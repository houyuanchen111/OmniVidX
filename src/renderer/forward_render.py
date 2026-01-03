import torch
import torch.nn.functional as F
import pyexr  
import torchvision
import numpy as np

try:
    from src.renderer.lighting import DirectionalLight_LatLong, GlobalIncidentLighting
    from src.renderer.render import IIR_RenderLayer
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from lighting import DirectionalLight_LatLong, GlobalIncidentLighting
    from render import IIR_RenderLayer

def run_rendering(albedo, roughness, metallic, normal, lighting_longitude, lighting_latitude, ambient=0.2, weight_init:float = 2):
    """
    Args:
        albedo: (3, H, W)
        roughness: (3, H, W)
        metallic: (3, H, W)
        normal: (3, H, W)
        lighting_longitude: (光源经度
        lighting_latitude: 光源纬度
        ambient: 控制整体的环境光亮度
        weight_init: 控制光源强度
    """
    def tonemap(image):
        mu = 64
        return torch.log(1 + mu * image)/ torch.log(torch.tensor(1 + mu, device=image.device, dtype=image.dtype))
    
    # Prepare lighting
    device = albedo.device
    lighting_model = GlobalIncidentLighting(value=DirectionalLight_LatLong(weight_init=weight_init)).to(device)
    # Fix: Use .data to avoid gradient tracking issues with in-place operations on nn.Parameter
    lighting_model.value.theta.data[:] = lighting_latitude
    lighting_model.value.phi.data[:] = lighting_longitude

    # Prepare the renderer
    renderer = IIR_RenderLayer(imWidth=albedo.shape[2],
                               imHeight=albedo.shape[1],
                               brdf_type="disney",
                               double_sided=False,
                               use_specular=True).to(device)
    
    # Render the image
    colorDiffuse, colorSpec, wi_mask, shading = renderer(lighting_model=lighting_model,
                                                         albedo=albedo.unsqueeze(0),
                                                         rough=roughness.unsqueeze(0),
                                                         metal=metallic.unsqueeze(0),
                                                         normal=normal.unsqueeze(0))
    rendered_image = (1 - ambient) * (colorDiffuse + colorSpec) + ambient * albedo.unsqueeze(0).to(device)
    rendered_image = rendered_image[0]
    
    # Tonemapping
    rendered_image = tonemap(rendered_image.clamp(0, 1))
    # torchvision.utils.save_image(rendered_image, "rendered_image.png")
    return rendered_image


if __name__ == "__main__":
    # 特别挑一两个进行测试
    albedo = torch.from_numpy(pyexr.open("/share/project/cwm/houyuan.chen/diffusion_render_inverse_render/interiorverse_intrinsix_debug_data/L3D124S8ENDIDQ5QIAUI5NYALUF3P3XA888/000_albedo.exr").get()).permute(2, 0, 1).cuda()
    material = torch.from_numpy(pyexr.open("/share/project/cwm/houyuan.chen/diffusion_render_inverse_render/interiorverse_intrinsix_debug_data/L3D124S8ENDIDQ5QIAUI5NYALUF3P3XA888/000_material.exr").get()).permute(2, 0, 1).cuda()
    roughness = material[[0]].repeat(3, 1, 1)  
    metallic = material[[1]].repeat(3, 1, 1)
    normal = torch.from_numpy(pyexr.open("/share/project/cwm/houyuan.chen/diffusion_render_inverse_render/interiorverse_intrinsix_debug_data/L3D124S8ENDIDQ5QIAUI5NYALUF3P3XA888/000_normal.exr").get()).permute(2, 0, 1).cuda()
    run_rendering(albedo, roughness, metallic, normal, torch.tensor(0).cuda().to(torch.float32).requires_grad_(True), torch.tensor(np.pi/2).cuda().to(torch.float32).requires_grad_(True))
    run_rendering(albedo, roughness, metallic, normal, torch.tensor(0).cuda().to(torch.float32).requires_grad_(True), torch.tensor(np.pi/4).cuda().to(torch.float32).requires_grad_(True))
    run_rendering(albedo, roughness, metallic, normal, torch.tensor(0).cuda().to(torch.float32).requires_grad_(True), torch.tensor(np.pi/6).cuda().to(torch.float32).requires_grad_(True))
    run_rendering(albedo, roughness, metallic, normal, torch.tensor(np.pi/2).cuda().to(torch.float32).requires_grad_(True), torch.tensor(0).cuda().to(torch.float32).requires_grad_(True))
    run_rendering(albedo, roughness, metallic, normal, torch.tensor(np.pi/4).cuda().to(torch.float32).requires_grad_(True), torch.tensor(np.pi/2).cuda().to(torch.float32).requires_grad_(True))
    run_rendering(albedo, roughness, metallic, normal, torch.tensor(np.pi/6).cuda().to(torch.float32).requires_grad_(True), torch.tensor(np.pi/2).cuda().to(torch.float32).requires_grad_(True))