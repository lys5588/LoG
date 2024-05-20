from torch.utils.cpp_extension import load

compute_radius_module = load(
    verbose=True,
    name='compute_radius',
    sources=['LoG/cuda/compute_radius_kernel.cu'],
    extra_include_paths=['submodules/mydiffgaussian/third_party/glm'],
    extra_cuda_cflags=['-O2']
)

# radius2d_cuda = compute_radius_module.compute_radius(
#                 xyz, scaling, rotation,
#                 proj_matrix, view_matrix,
#                 focal_x, focal_y, tanfovx, tanfovy)