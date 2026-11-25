// apple-accelerate-matmul on vLLM's Triton attention (kernel_unified_attention,
// 2-D prefill: fp16, head size 128, 32 query heads, 4 per KV head, KV blocks
// of 16, tiles of 32). Both dots (Q.K^T and P.V) take the simdgroup_matrix
// layout with all 4 SIMD-groups along N. The online softmax stays in it: the
// row max and row sum are slices of it, and the loop carries them and the
// accumulator without conversions. P reaches the second dot through
// threadgroup memory, and the only conversion left is before the store.
//
// REQUIRES: vllm
// RUN: %triton_to_ttgir %vllm_dir/v1/attention/ops/triton_unified_attention.py \
// RUN:   kernel_unified_attention --vllm --plugin %apple_triton_plugin \
// RUN:   --signature 'output_ptr=*fp16,query_ptr=*fp16,key_cache_ptr=*fp16,value_cache_ptr=*fp16,block_tables_ptr=*i32,seq_lens_ptr=*i32,scale=fp32,q_scale=*fp32,k_scale=*fp32,v_scale=*fp32,out_scale=fp32,softcap=fp32,block_table_stride=i64,query_stride_0=i64,query_stride_1=i64,output_stride_0=i64,output_stride_1=i64,qq_bias_stride_0=i64,rswa_prefix_lens_ptr=*i32,stride_k_cache_0=i64,stride_k_cache_1=i64,stride_k_cache_2=i64,stride_v_cache_0=i64,stride_v_cache_1=i64,stride_v_cache_2=i64,query_start_len_ptr=*i32,num_seqs=i32' \
// RUN:   --constexpr 'sink_ptr=None,alibi_slopes_ptr=None,qq_bias_ptr=None,per_seq_causal_ptr=None,mm_prefix_range_ptr=None,num_query_heads=32,num_queries_per_kv=4,BLOCK_SIZE=16,TILE_SIZE=32,HEAD_SIZE=128,HEAD_SIZE_PADDED=128,USE_ALIBI_SLOPES=False,USE_ALIBI_SQRT=False,USE_QQ_BIAS=False,USE_SOFTCAP=False,USE_SINKS=False,SLIDING_WINDOW=0,USE_CAUSAL=True,USE_PER_SEQ_CAUSAL=False,USE_MM_PREFIX=False,MAX_MM_RANGES=0,R_SWA_WINDOW=0,USE_R_SWA=False,stride_k_cache_3=1,stride_v_cache_3=1,BLOCK_Q=4,BLOCK_M=16,NUM_SEGMENTS_PER_SEQ=1,USE_FP8=False,IS_3D=False' \
// RUN:   2>/dev/null > %t.ttgir
// RUN: FileCheck %s < %t.ttgir
// RUN: grep -c '"ttg.convert_layout"' %t.ttgir | FileCheck %s --check-prefix=CONVERSIONS

// CHECK: "tt.dot"{{.*}} -> tensor<16x32xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>>
// CHECK: "math.exp"{{.*}} : (tensor<16x32xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>>)
// CHECK: "ttg.local_alloc"{{.*}} : (tensor<16x32xf16, #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>>)
// CHECK: "ttg.local_load"{{.*}} -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>}>>
// CHECK: "tt.dot"{{.*}} -> tensor<16x128xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>>
// CHECK: "scf.yield"({{.*}}) : (tensor<16xf32, #ttg.slice<{dim = 1, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>}>>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>}>>, tensor<16x128xf32, #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>>)
// CHECK: "ttg.convert_layout"{{.*}} : (tensor<16x128xf16, #apple_tt.simdgroup_mma<{warpsPerCTA = [1, 4]}>>) -> tensor<16x128xf16, #ttg.blocked
// CHECK: "tt.store"
// CONVERSIONS: 1
