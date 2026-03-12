from dataset.mri_code import build_mri_code


def build_dataset(args, **kwargs):
    if args.dataset == 'mri_code':
        return build_mri_code(args, **kwargs)
    raise NotImplementedError(f'Dataset {args.dataset} is not supported')
