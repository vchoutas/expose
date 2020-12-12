# pylint: disable=too-many-branches, too-many-statements
import argparse

from openpifpaf.network import nets
from openpifpaf import decoder


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Subparser definition
    subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')
    predict_parser = subparsers.add_parser("predict")
    prep_parser = subparsers.add_parser("prep")
    training_parser = subparsers.add_parser("train")
    eval_parser = subparsers.add_parser("eval")

    # Preprocess input data
    prep_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints', required=True)
    prep_parser.add_argument('--dataset',
                             help='datasets to preprocess: nuscenes, nuscenes_teaser, nuscenes_mini, kitti',
                             default='nuscenes')
    prep_parser.add_argument('--dir_nuscenes', help='directory of nuscenes devkit', default='data/nuscenes/')
    prep_parser.add_argument('--iou_min', help='minimum iou to match ground truth', type=float, default=0.3)

    # Predict (2D pose and/or 3D location from images)
    # General
    predict_parser.add_argument('--networks', nargs='+', help='Run pifpaf and/or monoloco', default=['monoloco'])
    predict_parser.add_argument('images', nargs='*', help='input images')
    predict_parser.add_argument('--glob', help='glob expression for input images (for many images)')
    predict_parser.add_argument('-o', '--output-directory', help='Output directory')
    predict_parser.add_argument('--output_types', nargs='+', default=['json'],
                                help='what to output: json keypoints skeleton for Pifpaf'
                                     'json bird front combined for Monoloco')
    predict_parser.add_argument('--show', help='to show images', action='store_true')

    # Pifpaf
    nets.cli(predict_parser)
    decoder.cli(predict_parser, force_complete_pose=True, instance_threshold=0.15)
    predict_parser.add_argument('--scale', default=1.0, type=float, help='change the scale of the image to preprocess')

    # Monoloco
    predict_parser.add_argument('--model', help='path of MonoLoco model to load', required=True)
    predict_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=512)
    predict_parser.add_argument('--path_gt', help='path of json file with gt 3d localization',
                                default='data/arrays/names-kitti-190513-1754.json')
    predict_parser.add_argument('--transform', help='transformation for the pose', default='None')
    predict_parser.add_argument('--draw_box', help='to draw box in the images', action='store_true')
    predict_parser.add_argument('--predict', help='whether to make prediction', action='store_true')
    predict_parser.add_argument('--z_max', type=int, help='maximum meters distance for predictions', default=22)
    predict_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    predict_parser.add_argument('--dropout', type=float, help='dropout parameter', default=0.2)
    predict_parser.add_argument('--webcam', help='monoloco streaming', action='store_true')

    # Training
    training_parser.add_argument('--joints', help='Json file with input joints',
                                 default='data/arrays/joints-nuscenes_teaser-190513-1846.json')
    training_parser.add_argument('--save', help='whether to not save model and log file', action='store_false')
    training_parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train for', default=150)
    training_parser.add_argument('--bs', type=int, default=256, help='input batch size')
    training_parser.add_argument('--baseline', help='whether to train using the baseline', action='store_true')
    training_parser.add_argument('--dropout', type=float, help='dropout. Default no dropout', default=0.2)
    training_parser.add_argument('--lr', type=float, help='learning rate', default=0.002)
    training_parser.add_argument('--sched_step', type=float, help='scheduler step time (epochs)', default=20)
    training_parser.add_argument('--sched_gamma', type=float, help='Scheduler multiplication every step', default=0.9)
    training_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=256)
    training_parser.add_argument('--n_stage', type=int, help='Number of stages in the model', default=3)
    training_parser.add_argument('--hyp', help='run hyperparameters tuning', action='store_true')
    training_parser.add_argument('--multiplier', type=int, help='Size of the grid of hyp search', default=1)
    training_parser.add_argument('--r_seed', type=int, help='specify the seed for training and hyp tuning', default=1)

    # Evaluation
    eval_parser.add_argument('--dataset', help='datasets to evaluate, kitti or nuscenes', default='kitti')
    eval_parser.add_argument('--geometric', help='to evaluate geometric distance', action='store_true')
    eval_parser.add_argument('--generate', help='create txt files for KITTI evaluation', action='store_true')
    eval_parser.add_argument('--dir_ann', help='directory of annotations of 2d joints (for KITTI evaluation')
    eval_parser.add_argument('--model', help='path of MonoLoco model to load', required=True)
    eval_parser.add_argument('--joints', help='Json file with input joints to evaluate (for nuScenes evaluation)')
    eval_parser.add_argument('--n_dropout', type=int, help='Epistemic uncertainty evaluation', default=0)
    eval_parser.add_argument('--dropout', type=float, help='dropout. Default no dropout', default=0.2)
    eval_parser.add_argument('--hidden_size', type=int, help='Number of hidden units in the model', default=256)
    eval_parser.add_argument('--n_stage', type=int, help='Number of stages in the model', default=3)
    eval_parser.add_argument('--show', help='whether to show statistic graphs', action='store_true')
    eval_parser.add_argument('--save', help='whether to save statistic graphs', action='store_true')
    eval_parser.add_argument('--verbose', help='verbosity of statistics', action='store_true')
    eval_parser.add_argument('--stereo', help='include stereo baseline results', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = cli()
    if args.command == 'predict':
        if args.webcam:
            from .visuals.webcam import webcam
            webcam(args)
        else:
            from . import predict
            predict.predict(args)

    elif args.command == 'prep':
        if 'nuscenes' in args.dataset:
            from .prep.preprocess_nu import PreprocessNuscenes
            prep = PreprocessNuscenes(args.dir_ann, args.dir_nuscenes, args.dataset, args.iou_min)
            prep.run()
        if 'kitti' in args.dataset:
            from .prep.preprocess_ki import PreprocessKitti
            prep = PreprocessKitti(args.dir_ann, args.iou_min)
            prep.run()

    elif args.command == 'train':
        from .train import HypTuning
        if args.hyp:
            hyp_tuning = HypTuning(joints=args.joints, epochs=args.epochs,
                                   baseline=args.baseline, dropout=args.dropout,
                                   multiplier=args.multiplier, r_seed=args.r_seed)
            hyp_tuning.train()
        else:
            from .train import Trainer
            training = Trainer(joints=args.joints, epochs=args.epochs, bs=args.bs,
                               baseline=args.baseline, dropout=args.dropout, lr=args.lr, sched_step=args.sched_step,
                               n_stage=args.n_stage, sched_gamma=args.sched_gamma, hidden_size=args.hidden_size,
                               r_seed=args.r_seed, save=args.save)

            _ = training.train()
            _ = training.evaluate()

    elif args.command == 'eval':
        if args.geometric:
            assert args.joints, "joints argument not provided"
            from .eval import geometric_baseline
            geometric_baseline(args.joints)

        if args.generate:
            from .eval import GenerateKitti
            kitti_txt = GenerateKitti(args.model, args.dir_ann, p_dropout=args.dropout, n_dropout=args.n_dropout,
                                      stereo=args.stereo)
            kitti_txt.run()

        if args.dataset == 'kitti':
            from .eval import EvalKitti
            kitti_eval = EvalKitti(verbose=args.verbose, stereo=args.stereo)
            kitti_eval.run()
            kitti_eval.printer(show=args.show, save=args.save)

        if 'nuscenes' in args.dataset:
            from .train import Trainer
            training = Trainer(joints=args.joints)
            _ = training.evaluate(load=True, model=args.model, debug=False)

    else:
        raise ValueError("Main subparser not recognized or not provided")


if __name__ == '__main__':
    main()
