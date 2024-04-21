from advertorch.attacks import *


def make_img_atk(
    args,
    net,
    atk_eps,
):
    obs_atk = None

    if args.atk_type in ["fgm", "fgsm", "GradientSignAttack"]:
        obs_atk = GradientSignAttack(
            net,
            eps=atk_eps,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            targeted=args.targeted,
        )
    elif args.atk_type in ["cw", "CarliniWagnerL2Attack"]:
        obs_atk = CarliniWagnerL2Attack(
            net,
            args.act_shape,
            confidence=args.confidence,
            max_iterations=args.n_iter,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            targeted=args.targeted,
            learning_rate=args.learning_rate,
            binary_search_steps=args.binary_search_steps,
            abort_early=args.abort_early,
            initial_const=args.initial_const,
        )
    elif args.atk_type in ["pgda", "pgd", "PGDAttack", "LinfPGDAttack"]:
        obs_atk = PGDAttack(
            net,
            eps=atk_eps * args.max_pixel,
            targeted=args.targeted,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            nb_iter=args.n_iter,
            eps_iter=atk_eps * args.max_pixel,
            rand_init=args.rand_init,
            l1_sparsity=args.l1_sparsity
        )
    elif args.atk_type == "L2PGDAttack":
        obs_atk = L2PGDAttack(
            net,
            eps=atk_eps * args.max_pixel,
            targeted=args.targeted,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            nb_iter=args.n_iter,
            eps_iter=atk_eps * args.max_pixel,
            rand_init=args.rand_init
        )
    elif args.atk_type == "SparseL1DescentAttack":
        obs_atk = SparseL1DescentAttack(
            net,
            eps=atk_eps * args.max_pixel,
            targeted=args.targeted,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            nb_iter=args.n_iter,
            eps_iter=atk_eps * args.max_pixel,
            rand_init=args.rand_init,
            l1_sparsity=args.l1_sparsity
        )
    elif args.atk_type in [
        "MomentumIterativeAttack",
        "LinfMomentumIterativeAttack",
    ]:
        obs_atk = MomentumIterativeAttack(
            net,
            eps=atk_eps * args.max_pixel,
            targeted=args.targeted,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            nb_iter=args.n_iter,
            eps_iter=atk_eps * args.max_pixel * 2 / args.n_iter,
        )
    elif args.atk_type == "ElasticNetL1Attack":
        obs_atk = ElasticNetL1Attack(
            net,
            args.action_shape,
            confidence=args.confidence,
            max_iterations=args.n_iter,
            targeted=args.targeted,
            clip_min=args.min_pixel,
            clip_max=args.max_pixel,
            learning_rate=args.learning_rate,
            binary_search_steps=args.binary_search_steps,
            abort_early=args.abort_early,
            initial_const=args.initial_const
        )
    else:
        raise Exception("Attack method not defined")
    return obs_atk
