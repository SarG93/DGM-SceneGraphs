from train import *
from generate import *
from evaluate import *



if __name__ == "__main__":
    print('Loading data..')
    graphs_all, ind_to_classes, ind_to_predicates = pickle.load(open(os.path.join('./data','new_graph_data_connected.p'), 'rb'))
    shuffle(graphs_all)
    graphs_train = graphs_all[:int(0.7*len(graphs_all))]
    graphs_test = graphs_all[int(0.7*len(graphs_all)):]
    print('Number of training samples: ', len(graphs_train))
    print('Number of test samples: ', len(graphs_test))

    # set hyperparameters
    hyperparams_list = [[] for _ in range(3)]
    hyperparams_list[0] = [64] # batch size
    hyperparams_list[1] = [0.001] # node initial lerning rate
    hyperparams_list[2] = [300] # number of epochs
    #hyperparams_list[3] = [16, 64, 256]
    hyperparams = list(itertools.product(*hyperparams_list))

    print('Begin model tuning')
    model_dict = {}
    run_str = 'beta1_lsize64_1sample_without_coupling_'
    for hyperparam in hyperparams:
        batch_size, lr, num_epochs = hyperparam
        # set parameters
        # print('Setting model parameters..')
        params = Model_params_VAE(batch_size, lr, num_epochs)
        hyperparam_str = 'batch-'+str(batch_size) + '_epochs-'+str(num_epochs) + '_lr-'+str(params.lr)
        # instantiate models
        models = instantiate_setVAE_models(params)
        dataloader = create_dataloader_VAE(params, graphs_train, ind_to_classes)
        # train model
        trained_model = train_setVAE(params, dataloader, models, './models', run_str+hyperparam_str)
        # generate nodes
        #params, trained_model = load_VAE_model('./models', run_str+hyperparam_str)
        gen_data = generate_objects_VAE(params, trained_model, 20000)
        gen_path = os.path.join('./generated_samples', run_str+hyperparam_str)
        os.makedirs(gen_path, exist_ok=True)
        pickle.dump( gen_data, open(os.path.join(gen_path, 'generated_objects.p'), 'wb'))
