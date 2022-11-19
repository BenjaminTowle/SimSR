from src.modeling import biencoder, crossencoder


def get_model(args):
    models = {
        "biencoder": biencoder.DistilBertBiencoder,
        "crossencoder": crossencoder.BinaryCrossEncoder,
    }

    return models[args.model_type].from_pretrained(args.bert_model_path)
