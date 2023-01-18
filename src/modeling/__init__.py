from src.modeling import biencoder

def _get_matching(args):
    return biencoder.DistilBertBiencoder.from_pretrained(
        args.bert_model_path, use_symmetric_loss=args.use_symmetric_loss)

def _get_mcvae(args):
    return biencoder.DistilBertCVAE.from_pretrained(
        args.bert_model_path, 
        use_symmetric_loss=args.use_symmetric_loss, 
        z=args.z,
        kld_weight=args.kld_weight,
        use_kld_annealling=args.use_kld_annealling,
        kld_annealling_steps=args.kld_annealling_steps,
        use_message_prior=args.use_message_prior
    )


def get_model(args):
    models = {
        "matching": _get_matching,
        "mcvae": _get_mcvae
    }

    return models[args.model_type](args)
