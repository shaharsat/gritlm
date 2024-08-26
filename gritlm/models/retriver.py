from EasyLM.utils import distributed_utils

from typing import Optional, Union
from dataclasses import dataclass, field, asdict
from typing import Optional

from EasyLM.jax_utils import (
    with_sharding_constraint, get_jax_mesh, get_gradient_checkpoint_policy, put_along_zeroth_axis, create_target_scores,
    add_process_dim, remove_process_dim
)
from functools import partial

from EasyLM.modules.rpt_utils import (
    RetrieverSupervision, breakpoint_if_nonfinite, _ranksigrelu, ranksigrelu, exists, default, create_jax_coor_descent,
    m1_cosine_decay_schedule, m1_linear_decay_schedule, topk_chunks,
    compute_pairs, compute_retrieval_metrics, masked_top_k, truncate_w_topk, create_topk_softmax, batch_take_along_axis,
    compute_ndcg_lambda,
    make_cross_mask, chunk_causal_pad, chunk_causal_unpad, tree_unstack, EncodedNeighbors, create_segment_mask,
    create_segment_mask_w_future, create_segment_mask_w_future_v2, create_segment_mask_w_future_v3
)
from EasyLM.modules.rpt_utils import RetrieverNeighborOutput, RetrieverEncodedOutput, RetrieverLossOutput

from EasyLM.modules.cca import CrossAttention

from flax.linen import combine_masks
import operator
import jax
import flax.linen as nn
import jax.numpy as jnp
import einops
import optax
from flax.linen import partitioning as nn_partitioning
import numpy as np

remat = nn_partitioning.remat

from EasyLM.modules.rpt2_config import RPT2Config


@dataclass
class RetrievalConfig:
    pooling_size: int = 64
    n_skip_chunks: Optional[int] = 32
    num_neighbors: int = 2
    is_lo_encoded_output: bool = False
    query_chunk_index: int = -2
    repeat_db: bool = False
    cca_only_on_last_n_chunks: Optional[int] = None
    disable_dist_matmul: bool = True
    batch_size: Optional[int] = None
    flatten_db: bool = False
    is_generating: bool = False
    key_pooling_size: int = 64
    allow_cache: bool = True
    remove_pcw_bos: bool = False
    zero_out_pcw: bool = False

    @staticmethod
    def process_input_ids(input_ids, as_dict=False, **kwargs):
        retrieval_config = RetrievalConfig(**kwargs)

        if retrieval_config.batch_size is None:
            retrieval_config.batch_size = input_ids.shape[0]

        if as_dict:
            retrieval_config = asdict(retrieval_config)

        return retrieval_config


class Retriever(nn.Module):
    config: RPT2Config
    dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None
    param_dtype: Optional[jnp.dtype] = jnp.float32

    def setup(self):
        attention_module = CrossAttention
        if self.config.remat_attention != '':
            attention_module = remat(
                attention_module, static_argnums=(1, 2, 3, 5, 6, 7),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention)

            )
        self.preret_bidir_attention = attention_module(self.config,
                                                       dtype=self.dtype,
                                                       precision=self.precision,
                                                       param_dtype=self.param_dtype,
                                                       is_cross_attention=True)
        self.preret_bi_attention_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype,
                                                     scale_init=jax.nn.initializers.constant(
                                                         self.config.cca_layernorm_init_scale),
                                                     )
        self.pre_key_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                         dtype=self.dtype,
                                         scale_init=jax.nn.initializers.constant(self.config.cca_layernorm_init_scale),
                                         )
        self.key_projection = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.pre_query_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                           dtype=self.dtype,
                                           scale_init=jax.nn.initializers.constant(
                                               self.config.cca_layernorm_init_scale),
                                           )
        self.query_projection = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.learned_margin = self.param('learned_margin', jax.nn.initializers.constant(0), (1,))

        if self.config.ss_schedule_steps is not None and self.config.ss_schedule_steps > 0 and \
                self.config.scheduled_sampling_max_prob is not None \
                and self.config.scheduled_sampling_min_prob is not None \
                and self.has_rng("dropout"):
            self.ss_rng = self.make_rng("dropout")
            if self.config.ss_sch_type == "cosine":
                self.scheduled_sampling_schedule_fn = m1_cosine_decay_schedule(
                    decay_steps=self.config.ss_schedule_steps,
                    min_value=self.config.scheduled_sampling_min_prob,
                    max_value=self.config.scheduled_sampling_max_prob)
            elif self.config.ss_sch_type == "linear":
                self.scheduled_sampling_schedule_fn = m1_linear_decay_schedule(
                    decay_steps=self.config.ss_schedule_steps,
                    min_value=self.config.scheduled_sampling_min_prob,
                    max_value=self.config.scheduled_sampling_max_prob)
            else:
                raise ValueError(f"Invalid scheduled sampling schedule type: {self.config.ss_sch_type}")


        else:
            self.scheduled_sampling_schedule_fn = None
        if self.config.post_lookup_layernorn_init_scale > 0:
            self.post_lookup_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                                      dtype=self.dtype,
                                                      scale_init=jax.nn.initializers.constant(
                                                          self.config.post_lookup_layernorn_init_scale),
                                                      )
        else:
            self.post_lookup_layernorm = None

        if self.config.bottleneck_pre_lookup:
            bottleneck_pre_lookup_hidden_size = 128
            # bottleneck_pre_lookup_hidden_size=self.config.bottleneck_pre_lookup_hidden_size #TODO
            initializer_range = self.config.initializer_range
            # initializer_range = self.config.bottleneck_pre_lookup_init #TODO

            self.downproj = nn.Dense(
                bottleneck_pre_lookup_hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(stddev=initializer_range),
                precision=self.precision,
            )
            self.upproj = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=jax.nn.initializers.normal(stddev=initializer_range),
                precision=self.precision,
            )

    def __call__(
            self,
            encoded_output,
            retriever_supervision: RetrieverSupervision,
            train_step: Optional[int],
            deterministic: bool,
            n_skip_chunks: int,
            num_neighbors: int,
            is_lo_encoded_output: bool,
            repeat_db: bool,
            query_chunk_index: int,
            cca_only_on_last_n_chunks: Optional[int],
            disable_dist_matmul: bool,
            flatten_db: bool,
            is_generating: bool,
            **other_retrieval_kwargs
    ):
        del other_retrieval_kwargs
        assert num_neighbors is not None
        assert n_skip_chunks is not None

        orig_dtype = encoded_output["query"].query_chunks.dtype
        do_dist_matmul = False
        og_batch_size, og_n_queries_per_batch, emb_dim = encoded_output["query"].query_chunks.shape

        if flatten_db:
            encoded_output = jax.tree.map(lambda x: einops.rearrange(x, "(h b) c ... -> h (b c) ...", h=1),
                                          encoded_output)
        query_encoded_output = encoded_output["query"]
        key_encoded_output = encoded_output["key"]

        key_chunks = key_encoded_output.key_chunks.astype(jnp.float32)
        query_chunks = query_encoded_output.query_chunks.astype(jnp.float32)

        assert query_chunks.shape[0] == key_chunks.shape[0]
        assert query_chunks.shape[-1] == key_chunks.shape[-1]
        batch_size, n_queries_per_batch, emb_dim = query_chunks.shape
        _, n_keys_per_batch, _ = key_chunks.shape  # key_chunks is of shape (batch_size, n_keys_per_batch, emb_dim)

        if do_dist_matmul:
            assert flatten_db
            assert query_chunks.shape[0] == 1
            assert key_chunks.shape[0] == 1
            query_based_scores = distributed_utils.dist_matmul_tpu(query_chunks.squeeze(0), key_chunks.squeeze(0))
            query_based_scores = query_based_scores.reshape((batch_size, n_queries_per_batch, -1))
        else:
            query_based_scores = jnp.einsum('bqd,bkd->bqk', query_chunks, key_chunks)

        if n_skip_chunks == 0:
            chunk_mask = jnp.ones(query_based_scores.shape).astype(bool)
        else:
            assert query_chunks.shape[0] == 1
            chunk_mask = query_encoded_output.chunk_mask
            if self.config.seg_mask_type == "old":
                segment_mask = create_segment_mask(query_based_scores.shape[1], n_skip_chunks)
            elif self.config.seg_mask_type == "future":
                segment_mask = create_segment_mask_w_future(query_based_scores, og_batch_size)
                if self.config.use_pcw:
                    assert False, "Causes causal leakage..."
            elif self.config.seg_mask_type == "future_v2":
                segment_mask = create_segment_mask_w_future_v2(query_based_scores, og_batch_size)
                if self.config.use_pcw:
                    assert False, "Causes causal leakage..."
            elif self.config.seg_mask_type == "future_v3":
                segment_mask = create_segment_mask_w_future_v3(query_based_scores, og_batch_size)
            else:
                assert False, "Invalid seg_mask_type"
            from EasyLM.jax_utils import print_mask
            # print_mask(segment_mask,verbose=True)
            chunk_mask &= segment_mask[None, ...]

        @jax.vmap
        def batch_topk_chunks(scores, where):
            return topk_chunks(scores, num_candidates=num_neighbors, where=where)

        query_based_scores = jnp.where(chunk_mask, query_based_scores, -jnp.inf)
        query_score_based_idx = batch_topk_chunks(query_based_scores, chunk_mask)

        scaled_scores = self.apply_scaling(query_based_scores, num_neighbors, chunk_mask)
        scaled_scores = jnp.where(chunk_mask, scaled_scores, 0)
        query_att_scores, query_neighbor_mask = batch_take_along_axis(scaled_scores, chunk_mask, query_score_based_idx)
        self.sow('intermediates', f'query_based_scores', query_based_scores)
        self.sow('intermediates', f'scaled_scores', scaled_scores)
        self.sow('intermediates', f'query_att_scores', query_att_scores)

        if retriever_supervision is not None and self.config.seg_mask_type == "old":
            print("before retriever_output: ", jax.tree.map(jnp.shape, retriever_supervision), flush=True)
            retriever_supervision = jax.tree.map(
                lambda x: einops.rearrange(x, '(b nw) ... -> b nw ...', b=chunk_mask.shape[0]), retriever_supervision)
            print("after retriever_output: ", jax.tree.map(jnp.shape, retriever_supervision), flush=True)
            ret_loss_obj = self.compute_retriever_loss(query_based_scores,
                                                       retriever_supervision,
                                                       chunk_mask,
                                                       num_neighbors)
            target_score_based_idx = ret_loss_obj.target_score_based_idx
            aux_loss = ret_loss_obj.aux_loss
            ret_metrics = ret_loss_obj.ret_metrics
            target_att_scores, target_neighbor_mask = batch_take_along_axis(scaled_scores, chunk_mask,
                                                                            target_score_based_idx)
            if self.config.use_allowed_tar_mask:
                target_neighbor_mask = ret_loss_obj.target_neighbor_mask  # ?????

            top_nei_idx, nei_mask, att_scores = self.apply_scheduled_sampling(
                query_score_based_idx=query_score_based_idx,
                chunk_mask=query_neighbor_mask.astype(bool),
                target_score_based_idx=target_score_based_idx,
                target_neighbor_mask=target_neighbor_mask.astype(bool),
                train_step=train_step,
                query_att_scores=query_att_scores,
                target_att_scores=target_att_scores,
                deterministic=deterministic)
        elif (n_skip_chunks == 0 and self.config.ss_rdb and repeat_db and not deterministic):
            assert False
            n_ctxs = key_chunks.shape[-2] // query_chunks.shape[-2]
            batch_idxs = jnp.arange(query_chunks.shape[-2])
            key_batch_idxs = jnp.repeat(batch_idxs, n_ctxs)
            batch_mask = (batch_idxs[:, None] == key_batch_idxs[None, :])[None, :]

            _, same_batch_score_based_idx, _ = apply_compute_query_scores(query_chunks,
                                                                          key_chunks,
                                                                          batch_mask)
            same_batch_att_scores, same_batch_neighbor_mask = batch_take_along_axis(scaled_scores, chunk_mask,
                                                                                    same_batch_score_based_idx)

            top_nei_idx, nei_mask, att_scores = self.apply_scheduled_sampling(
                query_score_based_idx=query_score_based_idx,
                chunk_mask=query_neighbor_mask.astype(bool),
                target_score_based_idx=same_batch_score_based_idx,
                target_neighbor_mask=same_batch_neighbor_mask.astype(bool),
                train_step=train_step,
                query_att_scores=query_att_scores,
                target_att_scores=same_batch_att_scores,
                deterministic=deterministic)
            aux_loss = None
            ret_metrics = {}

        else:
            top_nei_idx, nei_mask, att_scores = query_score_based_idx, query_neighbor_mask.astype(
                bool), query_att_scores
            aux_loss = None
            ret_metrics = {}

        att_scores = jnp.where(att_scores > 0, att_scores, 0).astype(orig_dtype)

        if self.config.stop_grad_trick:
            att_scores = att_scores - jax.lax.stop_gradient(att_scores)
        if self.config.prop_method == "add":
            att_scores = att_scores + 0.5
        elif self.config.prop_method in ["mult_k", "mult_kv"]:
            att_scores = att_scores + 1.0
        else:
            raise ValueError(f"prop_method {self.config.prop_method} not recognized")

        att_scores = optax.scale_gradient(att_scores, self.config.att_score_grad_mod)

        cand_hidden_states = key_encoded_output.encoded_hidden_states
        cand_attention_mask = key_encoded_output.attention_mask.astype(bool).reshape(cand_hidden_states.shape[:-1])

        seq_len = cand_hidden_states.shape[-2]
        # cand_hidden_states has shape (batch_size, n_keys_per_batch,  seq_len, emb_dim or 128)
        # cand_attention_mask has shape (batch_size, n_keys_per_batch,  seq_len)
        assert cand_hidden_states.shape[0] == batch_size
        assert cand_hidden_states.shape[1] == n_keys_per_batch
        folded_top_nei_idx = top_nei_idx.reshape(batch_size, -1, 1)

        if disable_dist_matmul:
            neighbor_hidden_states = jnp.take_along_axis(cand_hidden_states, folded_top_nei_idx[..., None], axis=1)
            neighbor_mask = jnp.take_along_axis(cand_attention_mask, folded_top_nei_idx, axis=1)
        else:
            folded_top_nei_idx = folded_top_nei_idx.reshape([-1, num_neighbors])
            cand_hidden_states = einops.rearrange(cand_hidden_states, 'b n s d -> (b n) s d')
            cand_attention_mask = einops.rearrange(cand_attention_mask, 'b n s -> (b n) s')
            n_proc = jax.lax.psum(1, axis_name="process")
            proc_indices = folded_top_nei_idx // n_proc
            local_indices = folded_top_nei_idx % n_proc
            print(cand_hidden_states.shape)
            print(proc_indices.shape)
            print(local_indices.shape, flush=True)

            neighbor_hidden_states, neighbor_mask = distributed_utils.tree_dist_lookup_tpu(
                (cand_hidden_states, cand_attention_mask.astype(jnp.float32)),
                proc_indices, local_indices)

            neighbor_mask = neighbor_mask.astype(bool)
        if self.config.bottleneck_pre_lookup:
            neighbor_hidden_states = self.upproj(neighbor_hidden_states)

        ##
        bqkr_shape = tuple([batch_size, n_queries_per_batch, num_neighbors, seq_len])
        neighbor_hidden_states = neighbor_hidden_states.reshape(bqkr_shape + (emb_dim,))
        neighbor_mask = neighbor_mask.reshape(bqkr_shape)
        att_scores = jnp.broadcast_to(att_scores[..., None], bqkr_shape)
        pre_nei_mask = jnp.broadcast_to(nei_mask[..., None], bqkr_shape)
        ##
        neighbor_mask = neighbor_mask & pre_nei_mask

        # att_scores = jnp.where(neighbor_mask, att_scores, 0) #possible fix
        nei_position_ids = jnp.clip(neighbor_mask.astype(jnp.int32).cumsum(axis=-1) - 1, a_min=0)

        if aux_loss is not None:
            if self.config.aux_scale > 0:
                loss_scale = self.config.aux_scale
            else:
                loss_scale = 0
                aux_loss = jax.lax.stop_gradient(aux_loss)
        else:
            loss_scale = None

        if self.post_lookup_layernorm is not None:
            neighbor_hidden_states = self.post_lookup_layernorm(neighbor_hidden_states)

        retriever_output = RetrieverNeighborOutput(aux_loss=None,
                                                   neighbor_hidden_states=neighbor_hidden_states,
                                                   loss_scale=None,
                                                   neighbor_mask=neighbor_mask,
                                                   retrieval_metrics=None,
                                                   att_scores=att_scores,
                                                   encoded_output=encoded_output,
                                                   nei_position_ids=nei_position_ids,
                                                   query_based_scores=query_based_scores,
                                                   top_nei_idx=top_nei_idx,
                                                   )
        print("retriever_output: ", jax.tree.map(jnp.shape, retriever_output), flush=True)
        if flatten_db:
            retriever_output = jax.tree.map(lambda x: einops.rearrange(x, "h (b c) ... -> (h b) c ... ",
                                                                       b=og_batch_size),
                                            retriever_output)

        retriever_output = retriever_output.replace(aux_loss=aux_loss,
                                                    loss_scale=loss_scale,
                                                    retrieval_metrics=ret_metrics)

        return retriever_output

    def apply_scaling(self, query_based_scores, num_neighbors, chunk_mask):
        if self.config.use_coor_descent:
            if self.config.coor_descent_type == "reg":
                if self.config.coor_descent_mult > 1:
                    coor_descent = create_jax_coor_descent(n_iters=30, k=num_neighbors + 2)
                else:
                    coor_descent = create_jax_coor_descent(n_iters=30, k=num_neighbors)
                scaled_scores = coor_descent(query_based_scores * self.config.coor_descent_mult, chunk_mask)
            elif self.config.coor_descent_type == "softmax":
                topk_softmax = create_topk_softmax(
                    k=int(num_neighbors * 1.5) if not self.is_initializing() else query_based_scores.shape[0],
                    mult=self.config.topk_softmax_mult,
                    scale=self.config.topk_softmax_scale)
                scaled_scores = topk_softmax(query_based_scores, chunk_mask)
            else:
                assert False, "Invalid coor_descent_type"


        else:
            @partial(
                jax.vmap,
                in_axes=(0, 0, None, None, None, None),
            )
            def apply_ranksigrelu(scores, chunk_mask, score_temp, learned_margin, atsc_margin_min,
                                  assure_positive=True):
                scaled_scores = ranksigrelu(scores / score_temp,
                                            chunk_mask,
                                            margin=jax.nn.softplus(learned_margin),
                                            offset=atsc_margin_min,
                                            assure_positive=assure_positive)

                return scaled_scores

            scaled_scores, _ = apply_ranksigrelu(query_based_scores,
                                                 chunk_mask,
                                                 self.config.score_temp,
                                                 self.learned_margin,
                                                 self.config.atsc_margin_min,
                                                 self.config.assure_positive)
        return scaled_scores

    def _preret_encode(self,
                       hidden_states,
                       attention_mask,
                       deterministic,
                       retrieval_kwargs,
                       output_attentions: bool = False,
                       mode: str = "both"
                       ):
        assert "pooling_size" in retrieval_kwargs
        pooling_size = retrieval_kwargs["pooling_size"]
        if np.prod(attention_mask.shape) != np.prod(hidden_states.shape[:-1]):
            assert len(attention_mask.shape) == 2
            assert len(hidden_states.shape) == 3
            attention_mask = attention_mask[:, :hidden_states.shape[1]]
        print(f"{hidden_states.shape=}")
        print(f"{pooling_size=}", flush=True)
        batch_size_times_n_windows, window_length, dim = hidden_states.shape
        assert (
                           window_length % pooling_size) == 0, f"{pooling_size=} {window_length=}"  # batch_size_times_n_windows, window_length, dim

        batch_size = retrieval_kwargs["batch_size"]
        ignore_prefix = retrieval_kwargs.get("ignore_prefix", 0)
        assert ignore_prefix < pooling_size

        n_windows = batch_size_times_n_windows // batch_size

        n_chunks_per_window = window_length // pooling_size
        database_size = n_windows * n_chunks_per_window

        # add a chunk dimension
        original_hidden_states = hidden_states.reshape([-1, pooling_size, hidden_states.shape[-1]])
        attention_mask = attention_mask.reshape([-1, pooling_size])

        if ignore_prefix > 0:
            attention_mask = attention_mask[..., ignore_prefix:]
            original_hidden_states = original_hidden_states[..., ignore_prefix:, :]
            pooling_size = pooling_size - ignore_prefix

        if self.config.disable_new_retrieval_weights:
            encoded_hidden_states = original_hidden_states
            preret_attention = tuple()
            pooled_hidden_states = jnp.mean(encoded_hidden_states, axis=-2, where=attention_mask[..., None])
            key_chunks = pooled_hidden_states
            query_chunks = pooled_hidden_states
        else:
            # 1. apply bi-dir attention
            preret_bi_output = self.preret_bidir_attention(
                self.preret_bi_attention_norm(original_hidden_states),  # hidden_states 0
                None,  # key_value_states 1
                None,  # position_ids 2
                None,  # kv_position_ids 3
                attention_mask,  # 4
                None,  # retriever_scores 5
                output_attentions,  # 6
                deterministic,  # 7
            )
            encoded_hidden_states = preret_bi_output[0] + original_hidden_states
            preret_attention = preret_bi_output[1:]
            # 2. pool
            pooled_hidden_states = jnp.mean(encoded_hidden_states, axis=-2, where=attention_mask[..., None])
            # 3. project to query chunks and key chunks
            key_chunks = self.key_projection(self.pre_key_norm(pooled_hidden_states))
            query_chunks = self.query_projection(self.pre_query_norm(pooled_hidden_states))

            chunk_mask = attention_mask.astype(bool).any(-1).reshape([batch_size, database_size, 1])
            encoded_hidden_states = original_hidden_states if self.config.use_pcw else encoded_hidden_states
            encoded_hidden_states = encoded_hidden_states.reshape([batch_size, database_size, pooling_size, dim])
            attention_mask = attention_mask.reshape(encoded_hidden_states.shape[:-1])
            if mode in ["key", "both"]:
                key_chunks = key_chunks / jnp.linalg.norm(key_chunks, axis=-1, keepdims=True)
                key_chunks = key_chunks.reshape([batch_size, database_size, dim])
            else:
                key_chunks = None

            if mode in ["query", "both"]:
                query_chunks = query_chunks / jnp.linalg.norm(query_chunks, axis=-1, keepdims=True)
                query_chunks = query_chunks.reshape([batch_size, database_size, dim])
            else:
                query_chunks = None

        if self.config.bottleneck_pre_lookup:
            encoded_hidden_states = self.downproj(encoded_hidden_states)

        return RetrieverEncodedOutput(
            encoded_hidden_states=encoded_hidden_states,
            attention_mask=attention_mask,
            key_chunks=key_chunks,
            query_chunks=query_chunks,
            chunk_mask=chunk_mask,
            preret_attention=preret_attention,
        )

    def preret_encode(self,
                      hidden_states,
                      attention_mask,
                      deterministic,
                      retrieval_kwargs,
                      output_attentions: bool = False,
                      ):
        pooling_size = retrieval_kwargs.get("pooling_size")
        key_pooling_size = retrieval_kwargs.get("key_pooling_size", pooling_size)

        if key_pooling_size == pooling_size:
            encoded_output = self._preret_encode(
                hidden_states,
                attention_mask,
                mode="both",
                deterministic=deterministic,
                retrieval_kwargs=retrieval_kwargs,
                output_attentions=output_attentions,
            )
            result = RetrieverEncodedOutput(
                encoded_hidden_states=encoded_output.encoded_hidden_states,
                attention_mask=encoded_output.attention_mask,
                key_chunks=encoded_output.key_chunks,
                query_chunks=encoded_output.query_chunks,
                chunk_mask=encoded_output.chunk_mask,
                preret_attention=encoded_output.preret_attention,
            )
            return dict(key=result, query=result)
        else:
            key_encoded_output = self._preret_encode(
                hidden_states,
                attention_mask,
                mode="key",
                deterministic=deterministic,
                retrieval_kwargs={**retrieval_kwargs, "pooling_size": key_pooling_size},
                output_attentions=output_attentions,
            )

            query_encoded_output = self._preret_encode(
                hidden_states,
                attention_mask,
                mode="query",
                deterministic=deterministic,
                retrieval_kwargs=retrieval_kwargs,
                output_attentions=output_attentions,
            )

            return dict(key=key_encoded_output, query=query_encoded_output)

    def apply_scheduled_sampling(self,
                                 query_score_based_idx,
                                 chunk_mask,
                                 target_score_based_idx,
                                 target_neighbor_mask,
                                 train_step,
                                 query_att_scores,
                                 target_att_scores,
                                 deterministic):
        if deterministic or self.is_initializing() or target_score_based_idx is None or self.scheduled_sampling_schedule_fn is None:
            top_nei_idx, top_nei_mask, top_att_scores = query_score_based_idx, chunk_mask, query_att_scores
        else:
            if self.config.lowres_ss:
                # n_doc_chunks =
                selector = jax.random.bernoulli(key=self.ss_rng,
                                                p=self.scheduled_sampling_schedule_fn(
                                                    train_step if not self.is_initializing() else 1),
                                                shape=tuple(query_score_based_idx.shape[:-1]) + (1,)
                                                # shape=(n_doc_chunks,1)
                                                ).astype(bool)
                top_nei_idx = jnp.where(selector, query_score_based_idx, target_score_based_idx)
                top_nei_mask = jnp.where(selector, chunk_mask, target_neighbor_mask)
                top_att_scores = jnp.where(selector, query_att_scores, target_att_scores)
            else:
                rv = jax.random.bernoulli(key=self.ss_rng,
                                          p=self.scheduled_sampling_schedule_fn(
                                              train_step if not self.is_initializing() else 1),
                                          shape=())  # this is a boolean of shape [1]
                top_nei_idx, top_nei_mask, top_att_scores = jax.lax.cond(rv,
                                                                         (), lambda args: (
                    query_score_based_idx, chunk_mask, query_att_scores),
                                                                         (), lambda args: (
                    target_score_based_idx, target_neighbor_mask, target_att_scores)
                                                                         )
        return top_nei_idx, top_nei_mask, top_att_scores

    @partial(
        nn.vmap,
        in_axes=(0, 0, 0, None),
        out_axes=0,
        variable_axes={'params': None, 'intermediates': 0, "cache": 0},
        split_rngs={'dropout': True, "params": False},
    )
    def compute_retriever_loss(self, raw_query_scores, retriever_supervision, chunk_mask, num_neighbors):
        raw_query_scores = raw_query_scores.astype(jnp.float32)

        eps = 1e-2

        def f(x):
            return x.reshape((-1, self.config.num_scored_neighbors))

        retriever_supervision = jax.tree_map(f, retriever_supervision)

        nei_idx = retriever_supervision.nei_idx  # [num_sequence_chunks, num_scored_neighbors]
        nei_scores = retriever_supervision.nei_scores

        raw_target_scores = create_target_scores(raw_query_scores, nei_idx, nei_scores,
                                                 fill_value=self.config.retriever_fill_value)
        raw_target_scores_wz = create_target_scores(raw_query_scores, nei_idx, nei_scores, fill_value=0).astype(
            jnp.float32)

        threshold_mask = self.config.threshold_nei_scores < raw_target_scores
        allowed_neighbor_mask = combine_masks(threshold_mask, chunk_mask, dtype=bool)  # allowed neighbors

        pairs_diff = -compute_pairs(raw_query_scores, operator.sub)
        raw_target_scores = raw_target_scores / self.config.score_temp
        pairs_diff = pairs_diff / self.config.score_temp
        pair_loss = jax.nn.sigmoid(pairs_diff)

        # one of the scores needs to be above the threshold for the pair to be valid
        valid_pairs = combine_masks(compute_pairs(raw_target_scores, lambda x, y: x > y),
                                    compute_pairs(threshold_mask, lambda x, y: x),
                                    compute_pairs(chunk_mask, operator.and_)
                                    )
        any_mask = combine_masks(threshold_mask.any(axis=-1), chunk_mask.any(axis=-1), dtype=bool)
        ndcg_lambda = compute_ndcg_lambda(raw_query_scores, raw_target_scores_wz,
                                          query_mask=chunk_mask,
                                          target_mask=allowed_neighbor_mask, )

        pair_loss = jnp.where(valid_pairs, pair_loss, 0.0)

        metrics = compute_retrieval_metrics(raw_query_scores, raw_target_scores_wz,
                                            query_mask=chunk_mask,
                                            target_mask=allowed_neighbor_mask)
        metrics = jax.tree_map(lambda x: x.mean(), metrics)

        per_chunk_pair_loss = (ndcg_lambda * pair_loss).sum(axis=-1)

        raw_aux_loss = jnp.where(any_mask, per_chunk_pair_loss, 0.0).sum()

        target_idx = topk_chunks(raw_target_scores + raw_query_scores * eps, num_candidates=num_neighbors,
                                 where=chunk_mask)
        target_nei_mask = jnp.take_along_axis(allowed_neighbor_mask, target_idx, axis=-1)
        return RetrieverLossOutput(
            aux_loss=(raw_aux_loss, valid_pairs.sum(),),
            target_neighbor_mask=target_nei_mask,
            target_score_based_idx=target_idx,
            ret_metrics=metrics,
        )