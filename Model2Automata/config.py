
OUT_FOLDER = 'results/default'

INPUT_CLASSES = 50
OUTPUT_CLASSES = 15


def get_data_func(args):
    data_f, label = None, None

    if args.data == 'synt_data_rhlp':
        from data.rhlp_synt.data import get_train_test_data
        data_f = get_train_test_data

    return data_f


def get_base_model(args):
    base_m = None

    if args.no_hidden_model:
        if args.no_hidden_model == 'rhlp':
            from base_models.no_hidden_models.rhlp.Model import RHLPModel
            base_m = RHLPModel()
            base_m.load_model('base_models/no_hidden_models/rhlp/rhlp_6_5')

    else:
        if args.hidden_model == 'rhlp_hidden':
            from base_models.hidden_models.rhlp_hidden.Model import RHLPModel
            base_m = RHLPModel()
            base_m.load_model('base_models/hidden_models/rhlp_hidden/rhlp_6_5')

    return base_m


def get_clustering_method(args):
    x_disc, y_disc = None, None
    if args.x_clustering == 'k-means':
        from discrete_model.clustering_models import k_means
        x_disc = k_means.KMeansWrapper

    elif args.x_clustering == 'k-means-rep':
        from discrete_model.clustering_models import k_means_rep
        x_disc = k_means_rep.KMeansRepWrapper

    if args.prediction_clustering == 'k-means':
        from discrete_model.clustering_models import k_means
        y_disc = k_means.KMeansWrapper

    elif args.prediction_clustering == 'k-means-rep':
        from discrete_model.clustering_models import k_means_rep
        y_disc = k_means_rep.KMeansRepWrapper

    return x_disc, y_disc
