import numpy as np
from numpy.lib.npyio import save
from pyparsing import col
import torch
import sys
import os

import copy

import global_var
from util import *
from render import *

sys.path.insert(1, 'materialGAN/higan/models/')
from stylegan2_generator import StyleGAN2Generator


def loadLightAndCamera(in_dir):
    camera_pos = np.loadtxt(os.path.join(in_dir, 'camera_pos.txt'), delimiter=',').astype(np.float32)

    light_pos = np.loadtxt(os.path.join(in_dir, 'light_pos.txt'), delimiter=',').astype(np.float32)

    im_size = np.loadtxt(os.path.join(in_dir, 'image_size.txt'), delimiter=',')
    im_size = float(im_size)
    light = np.loadtxt(os.path.join(in_dir, 'light_power.txt'), delimiter=',')

    return light_pos[0], camera_pos[0], im_size, light


def save_image(output, save_dir, save_name, make_dir=False, is_naive=False):

    base_dir = save_dir

    if make_dir:
        save_dir = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    all_images_np = output['image']  # shape [num_interps, H, W, C]

    material_maps = []
    for i in range(len(all_images_np)):
        maps = all_images_np[i]  # shape [B, C, H, W]
        material_maps.append(maps)

    for material_map in material_maps:
        map_t = torch.from_numpy(material_map)
        map_t = torch.unsqueeze(map_t, 0)

        albedo, normal, roughness, specular = tex2map(map_t)

        albedo = gyTensor2Array(albedo[0, :].permute(1, 2, 0))
        normal = gyTensor2Array((normal[0, :].permute(1, 2, 0) + 1) / 2)
        roughness = gyTensor2Array(roughness[0, :].permute(1, 2, 0))
        specular = gyTensor2Array(specular[0, :].permute(1, 2, 0))

        albedo = gyArray2PIL(gyApplyGamma(albedo, 1 / 2.2))
        normal = gyArray2PIL(normal)
        roughness = gyArray2PIL(gyApplyGamma(roughness, 1 / 2.2))
        specular = gyArray2PIL(gyApplyGamma(specular, 1 / 2.2))

        specular.save(os.path.join(save_dir, 'specular.png'))
        albedo.save(os.path.join(save_dir, 'albedo.png'))
        normal.save(os.path.join(save_dir, 'normal.png'))
        roughness.save(os.path.join(save_dir, 'rough.png'))


def generateLightCameraPosition(p, angle, colocated=True, addNoise=True):
    theta = (np.pi/180 * np.array([0] + [angle]*8)).astype(np.float32)
    phi   = (np.pi/4   * np.array([0,1,5,3,7,2,6,4,8])).astype(np.float32)

    light_pos = np.stack((p * np.sin(theta) * np.cos(phi),
                          p * np.sin(theta) * np.sin(phi),
                          p * np.cos(theta))).transpose()
    if addNoise:
        light_pos[:,0:2] += np.random.randn(9,2).astype(np.float32)

    if colocated:
        camera_pos = light_pos.copy()
    else:
        camera_pos = np.array([[0,0,p]]).astype(np.float32).repeat(9,axis=0)
        if addNoise:
            camera_pos[:,0:2] += np.random.randn(9,2).astype(np.float32)
            
    return light_pos[0], camera_pos[0]


def save_render_and_map(save_name, save_dir, img_np, in_dir):

    light_position_np, camera_position_np, _, light_intensity_np = loadLightAndCamera(in_dir)


    with torch.no_grad():
        if th.cuda.is_available():
            light_intensity_t = torch.from_numpy(light_intensity_np).cuda()
        else:
            light_intensity_t = torch.from_numpy(light_intensity_np)

        img_t = torch.from_numpy(img_np)
        if th.cuda.is_available():
            img_t = torch.unsqueeze(img_t, 0).cuda()
        else:
            img_t = torch.unsqueeze(img_t, 0)

        albedo, normal, roughness, specular = tex2map(img_t)

        albedo = gyTensor2Array(albedo[0, :].permute(1, 2, 0))
        normal = gyTensor2Array((normal[0, :].permute(1, 2, 0) + 1) / 2)
        roughness = gyTensor2Array(roughness[0, :].permute(1, 2, 0))
        specular = gyTensor2Array(specular[0, :].permute(1, 2, 0))

        albedo = gyArray2PIL(gyApplyGamma(albedo, 1 / 2.2))
        normal = gyArray2PIL(normal)
        roughness = gyArray2PIL(gyApplyGamma(roughness, 1 / 2.2))
        specular = gyArray2PIL(gyApplyGamma(specular, 1 / 2.2))

        specular.save(os.path.join(save_dir, f'{save_name}_specular.png'))
        albedo.save(os.path.join(save_dir, f'{save_name}_albedo.png'))
        normal.save(os.path.join(save_dir, f'{save_name}_normal.png'))
        roughness.save(os.path.join(save_dir, f'{save_name}_rough.png'))

        microfacetObj = Microfacet(res=256, size=20)
        render = microfacetObj.eval(img_t, light_position_np, camera_position_np, light_intensity_t)
        render = render[0].detach().cpu().numpy()
        render = np.transpose(render, (1, 2, 0))
        render = gyApplyGamma(render, 1 / 2.2)
        render = gyArray2PIL(render)
        render.save(os.path.join(save_dir, f'{save_name}_render.png'))


def render_map(img_np):
    light_position_np, camera_position_np = generateLightCameraPosition(20, 20, True, False)
    light_intensity_np = np.array([1500.0, 1500.0, 1500.0])

    with torch.no_grad():
        if th.cuda.is_available():
            light_intensity_t = torch.from_numpy(light_intensity_np).cuda()
        else:
            light_intensity_t = torch.from_numpy(light_intensity_np)

        img_t = torch.from_numpy(img_np)

        if th.cuda.is_available():
            img_t = torch.unsqueeze(img_t, 0).cuda()
        else:
            img_t = torch.unsqueeze(img_t, 0)
        microfacetObj = Microfacet(res=256, size=20)
        render = microfacetObj.eval(img_t, light_position_np, camera_position_np, light_intensity_t)
        render = render[0].detach().cpu().numpy()
        render = np.transpose(render, (1, 2, 0))
        render = gyApplyGamma(render, 1 / 2.2)

        return np.clip(render, 0, 1)


def lerp_np(v0, v1, num_interps):
    if v0.shape != v1.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    alphas = np.linspace(0, 1, num_interps)
    return np.array([(1-a)*v0 + a*v1 for a in alphas])


def lerp(v0, v1, t):
    return (1-t)*v0 + t*v1


def lerp_noises(noises1, noises2, t):
    noises = []
    for noise1, noise2 in zip(noises1, noises2):
        noise_lerp = torch.lerp(noise1, noise2, t)
        noises.append(noise_lerp)
    return noises


def slerp(v0, v1, num_interps):
    """Spherical linear interpolation."""
    # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
    t_array = np.arange(0, 1, 1/num_interps)
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)

    if dot < 0.0:
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t_array
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])


def interpolate_random_textures(num_maps,
                                save_dir,
                                num_interps=10,
                                resolution=256,
                                noise_init="random",
                                search_type="brute",
                                interp_type="lerp",
                                latent_space_type='z'):
    global_var.init_global_noise(resolution, noise_init)
    genObj = StyleGAN2Generator('svbrdf')

    if search_type == "brute":

        z = genObj.sample(num_maps)
        w = genObj.synthesize(z)['z']

        nearest_neighbor_idx = [-1] * (len(w) - 1)
        for i in range(0, len(w)-1, 1):
            dist = sys.float_info.max
            for j in range(i + 1, len(w), 1):
                new_dist = np.linalg.norm(w[i] - w[j])
                if new_dist < dist:
                    nearest_neighbor_idx[i] = j

        # (n*(n-1)) / 2 pairs
        for i in range(0, len(nearest_neighbor_idx), 1):
            j = nearest_neighbor_idx[i]
            if interp_type == "lerp":
                interp = lerp_np(w[i], w[j], num_interps=num_interps)
            elif interp_type == "slerp":
                interp = slerp(w[i], w[j], num_interps=num_interps)
            outputs = genObj.synthesize(interp, latent_space_type='z')

            # save images
            save_name = "{}_{}".format(i, j)
            save_image(outputs, save_dir, save_name)

    elif search_type == "random":

        z0 = genObj.sample(num_maps)
        w0 = genObj.synthesize(z0)['w']

        z1 = genObj.sample(num_maps)
        w1 = genObj.synthesize(z1)['w']

        sorted_dist = []
        for i in range(0, len(w0), 1):
            w1_sorted = sorted(w1, key=lambda e: np.linalg.norm(w0[i] - e))
            sorted_dist.append(w1_sorted)

        for i in range(len(w0)):
            w1 = sorted_dist[i]
            j = np.random.randint(0, len(w1) // 2) # random from nearest half
            interp = lerp_np(w0[i], w1[j], num_interps=num_interps)
            outputs = genObj.synthesize(interp, latent_space_type='w')

            # save images
            save_name = "{}_{}".format(i, j)
            save_image(outputs, save_dir, save_name)


def interpolate_projected_textures(latent_paths,
                                   noises_paths,
                                   image_paths,
                                   save_dir,
                                   num_interps=10,
                                   resolution=256,
                                   search_type="brute",
                                   num_maps=10):

    global_var.init_global_noise(resolution, "random")
    genObj = StyleGAN2Generator('svbrdf')

    all_latents, all_noises = [], []

    for latent_path in latent_paths:
        latent = torch.load(latent_path).detach().cpu().numpy()
        all_latents.append(latent)

    for noises_path in noises_paths:
        noises_list = torch.load(noises_path)
        noise_vars = []
        for noise in noises_list:
            if th.cuda.is_available():
                noise = noise.cuda()
            noise_vars.append(noise)
        all_noises.append(noise_vars)

    all_images = []
    for image_path in image_paths:
        map, _ = png2tex(image_path)
        map_np = map.detach().cpu().numpy()
        all_images.append(map_np[0])

    if search_type == "brute":

        ts = np.linspace(0., 1., num_interps)

        for i in range(0, len(all_latents) - 1, 1):

            name1 = os.path.basename(os.path.dirname(latent_paths[i]))

            for j in range(i + 1, len(all_latents), 1):

                name2 = os.path.basename(os.path.dirname(latent_paths[j]))

                lerp_outputs = []
                for t in ts:
                    noises = lerp_noises(all_noises[i], all_noises[j], t)
                    lerp_latent = lerp(all_latents[i], all_latents[j], t)

                    global_var.noises = noises
                    outputs = genObj.synthesize(lerp_latent, latent_space_type="wp")
                    lerp_outputs.append(outputs['image'])

                lerp_outputs = np.concatenate(lerp_outputs, 0)

                lerp_maps = lerp_np(all_images[i], all_images[j], num_interps=num_interps)

                # save images
                save_name = "{}_VERSUS_{}".format(name1, name2)
                save_image({'image': lerp_outputs}, save_dir, save_name, make_dir=True)
                save_image({'image': lerp_maps}, save_dir, save_name, make_dir=True, is_naive=True)

    elif search_type == "random":

        rand_z = genObj.sample(num_maps)

        def rand_noise():
            global_var.init_global_noise(resolution, "random")
            return copy.deepcopy(global_var.noises)
        rand_noises = [rand_noise() for _ in range(len(rand_z))]

        rand_latents = genObj.synthesize(rand_z, latent_space_type="z")['wp']
        ts = np.linspace(0., 1., num_interps)

        for i in range(0, len(all_latents) - 1, 1):

            name1 = os.path.basename(os.path.dirname(latent_paths[i]))

            for j in range(0, len(rand_z), 1):
                lerp_outputs = []
                for t in ts:
                    noises = lerp_noises(all_noises[i], rand_noises[j], t)

                    global_var.noises = noises

                    lerp_latent = lerp(all_latents[i][0], rand_latents[j], t)

                    lerp_latent = np.expand_dims(lerp_latent, 0)
                    outputs = genObj.synthesize(lerp_latent, latent_space_type="wp")
                    lerp_outputs.append(outputs['image'])

                lerp_outputs = np.concatenate(lerp_outputs, 0)

                # save images
                save_name = "{}_{}".format(name1, j)
                save_image({'image': lerp_outputs}, save_dir, save_name)


def interpolate_single_texture(latent_paths,
                                   noises_paths,
                                   save_dir,
                                   resolution=256):

    global_var.init_global_noise(resolution, "random")
    genObj = StyleGAN2Generator('svbrdf')

    all_latents, all_noises = [], []

    for latent_path in latent_paths:
        latent = torch.load(latent_path).detach().cpu().numpy()
        all_latents.append(latent)

    for noises_path in noises_paths:
        noises_list = torch.load(noises_path)
        noise_vars = []
        for noise in noises_list:
            if th.cuda.is_available():
                noise = noise.cuda()
            noise_vars.append(noise)
        all_noises.append(noise_vars)

    ts = [0.4, -0.4]

    name1 = os.path.basename(os.path.dirname(latent_paths[0]))

    input_path = os.path.abspath(os.path.join(latent_paths[0], os.pardir, os.pardir))
    input_path = os.path.join(input_path, 'input/')

    for sem_id in range(1, len(all_latents), 1):

        print("Generating semantic {}/8".format(sem_id))

        for col_id in range(len(ts)):
            t = ts[col_id]
            noises = lerp_noises(all_noises[0], all_noises[sem_id], t)
            lerp_latent = lerp(all_latents[0], all_latents[sem_id], t)

            global_var.noises = noises
            outputs = genObj.synthesize(lerp_latent, latent_space_type="wp")

            latent = torch.from_numpy(lerp_latent)

            save_render_and_map(f"{sem_id}_{col_id}", save_dir, outputs['image'][0], input_path)
            
            torch.save(latent, os.path.join(save_dir, f'{sem_id}_{col_id}_optim_latent.pt'))
            torch.save(global_var.noises, os.path.join(save_dir, f'{sem_id}_{col_id}_optim_noise.pt'))
