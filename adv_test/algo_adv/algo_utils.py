from adv_test.algo_adv.algo_adv.black_algo_adv.ZOOAttack import (
    ZerothOrderOptimizationAttack,
)
from adv_test.algo_adv.algo_adv.white_algo_adv.FGSM import GradientSignAttackT
from adv_test.algo_adv.algo_adv.white_algo_adv.PGD import (
    PGDAttackT,
    SparseL1DescentAttackT,
    L2PGDAttackT,
)
from adv_test.algo_adv.algo_adv.white_algo_adv.MomeIter import MomentumIterativeAttackT
from adv_test.algo_adv.algo_adv.white_algo_adv.CW import CarliniWagnerL2AttackT
from adv_test.algo_adv.algo_adv.white_algo_adv.EAD import ElasticNetL1AttackT

import torch


def make_img_atk(
    args,
    net,
    atk_eps,
    high,
    low,
):
    obs_atk = None

    high_max = torch.FloatTensor(high).to(args.device)
    low_min = torch.FloatTensor(low).to(args.device)

    if args.atk_type in ["fgm", "fgsm", "GradientSignAttack"]:
        obs_atk = GradientSignAttackT(
            net,
            eps=atk_eps,
            clip_min=low_min,
            clip_max=high_max,
            targeted=args.targeted,
        )
    elif args.atk_type in ["cw", "CarliniWagnerL2Attack"]:
        obs_atk = CarliniWagnerL2AttackT(
            net,
            args.act_shape,
            confidence=args.confidence,
            max_iterations=args.n_iter,
            clip_min=low_min,
            clip_max=high_max,
            targeted=args.targeted,
            binary_search_steps=args.binary_search_steps,
            abort_early=args.abort_early,
            initial_const=args.initial_const,
        )
    elif args.atk_type in ["pgda", "pgd", "PGDAttack", "LinfPGDAttack"]:
        obs_atk = PGDAttackT(
            net,
            eps=atk_eps,
            targeted=args.targeted,
            clip_min=low_min,
            clip_max=high_max,
            nb_iter=args.n_iter,
            eps_iter=atk_eps / 10,
            rand_init=args.rand_init,
            l1_sparsity=args.l1_sparsity,
        )
    elif args.atk_type == "L2PGDAttack":
        obs_atk = L2PGDAttackT(
            net,
            eps=atk_eps,
            targeted=args.targeted,
            clip_min=low_min,
            clip_max=high_max,
            nb_iter=args.n_iter,
            eps_iter=atk_eps / 10,
            rand_init=args.rand_init,
        )
    elif args.atk_type == "SparseL1DescentAttack":
        obs_atk = SparseL1DescentAttackT(
            net,
            eps=atk_eps * 100,
            targeted=args.targeted,
            clip_min=low_min,
            clip_max=high_max,
            nb_iter=args.n_iter,
            eps_iter=atk_eps * 100,
            rand_init=args.rand_init,
            l1_sparsity=args.l1_sparsity,
        )
    elif args.atk_type in [
        "MomentumIterativeAttack",
        "LinfMomentumIterativeAttack",
    ]:
        obs_atk = MomentumIterativeAttackT(
            net,
            eps=atk_eps * 10,
            targeted=args.targeted,
            clip_min=low_min,
            clip_max=high_max,
            nb_iter=args.n_iter,
            eps_iter=atk_eps * 10 * 2 / args.n_iter,
        )
    elif args.atk_type == "ElasticNetL1Attack":

        obs_atk = ElasticNetL1AttackT(
            net,
            args.act_shape,
            confidence=args.confidence,
            max_iterations=args.n_iter,
            targeted=args.targeted,
            clip_min=low_min,
            clip_max=high_max,
            learning_rate=args.learning_rate,
            binary_search_steps=args.binary_search_steps,
            abort_early=args.abort_early,
            initial_const=args.initial_const,
        )
    elif args.atk_type in ["ZOO", "zoo"]:
        obs_atk = ZerothOrderOptimizationAttack(
            net,
            loss_fn=None,
            eps=atk_eps,
            nb_samples=args.n_iter,
            delta=0.01,
            clip_min=low_min,
            clip_max=high_max,
            targeted=args.targeted,
        )
    else:
        raise Exception("Attack method not defined")
    return obs_atk
