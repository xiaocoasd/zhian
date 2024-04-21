from advertorch.attacks import *
from adv_test.adv_sample_gene.deepfool import DeepfoolLinfAttack

def make_img_atk(
        net, atk_eps, atk_type, env, iterations, min_pixel=0.0, max_pixel=255.0, targeted=False
):
    obs_atk = None

    if atk_type in ["fgm", "fgsm", "GradientSignAttack"]:
        obs_atk = GradientSignAttack(
            net,
            eps=atk_eps,
            clip_min=min_pixel,
            clip_max=max_pixel,
            targeted=targeted,
        )
    elif atk_type in ['cw', 'CarliniWagnerL2Attack']:
        obs_atk = CarliniWagnerL2Attack(
            net,
            env.action_space,
            confidence=0.1,
            max_iterations=iterations,
            clip_min=min_pixel,
            clip_max=max_pixel,
            targeted=targeted
        )
    elif atk_type in ["pgda","pgd","PGDAttack","LinfPGDAttack"]:
        obs_atk = PGDAttack(
            net,
            # eps=atk_eps * max_pixel,
            eps=atk_eps,
            targeted=targeted,
            clip_min=min_pixel,
            clip_max=max_pixel,
            nb_iter=iterations,
            eps_iter=atk_eps * max_pixel
        )
    elif atk_type =="L2PGDAttack":
        obs_atk = L2PGDAttack(
            net,
            # eps=atk_eps * max_pixel,
            eps=atk_eps,
            targeted=targeted,
            clip_min=min_pixel,
            clip_max=max_pixel,
            nb_iter=iterations,
            eps_iter=atk_eps * max_pixel)
    elif  atk_type== "SparseL1DescentAttack":
        obs_adv_atk = SparseL1DescentAttack(
            net,
            # eps=atk_eps * max_pixel,
            eps=atk_eps,
            targeted=targeted,
            clip_min=min_pixel,
            clip_max=max_pixel,
            nb_iter=iterations,
            eps_iter=atk_eps * max_pixel)
    elif atk_type in ["MomentumIterativeAttack", "LinfMomentumIterativeAttack"]:
        obs_adv_atk = MomentumIterativeAttack(
            net,
            # eps=atk_eps * max_pixel,
            eps=atk_eps,
            targeted=targeted,
            clip_min=min_pixel,
            clip_max=max_pixel,
            nb_iter=iterations,
            eps_iter=atk_eps * max_pixel * 2 / iterations)
    elif atk_type == "ElasticNetL1Attack":
        obs_adv_atk = ElasticNetL1Attack(
            net,
            env.action_shape,
            confidence=0.1,
            max_iterations=iterations,
            targeted=targeted,
            clip_min=min_pixel,
            clip_max=max_pixel)
    elif atk_type in ["DeepfoolLinfAttack"]:
        assert targeted is False, "Deepfool only supports untargeted attacks"
        obs_adv_atk = DeepfoolLinfAttack(
            net,
            # eps=atk_eps * max_pixel,
            eps=atk_eps,
            clip_min=min_pixel,
            clip_max=max_pixel,
            nb_iter=iterations,
            )
    else:
        raise Exception("Attack method not defined")
    return obs_atk
