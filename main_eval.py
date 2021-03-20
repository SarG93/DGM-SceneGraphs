from train import *
from generate import *
from evaluate import *

# setVAE or graphRNN
use_setVAE =False
num_graphs = 1000
num_eval = 1000
gen_data = True
if __name__ == "__main__":
    print('Loading data..')
    graphs_train = pickle.load(open(os.path.join('./data','train_dataset.p'), 'rb'))
    graphs_test = pickle.load(open(os.path.join('./data','test_dataset.p'), 'rb'))
    ind_to_classes, ind_to_predicates, _ = pickle.load(open(os.path.join('./data','categories.p'), 'rb'))
    print('Number of training samples: ', len(graphs_train))
    print('Number of test samples: ', len(graphs_test))
    if use_setVAE:
        print('Using setVAE+ghraphRNN model')
        # generate nodes
        #hyperparam_str = 'latent32_warmup100_beta5_batch-256_epochs-500_lr-0.0005'
        #hyperparam_str = 'latent32_warmup100_beta1_batch-64_epochs-500_lr-0.0002'
        hyperparam_str = 'beta1_lsize16_1sample_without_coupling_batch-64_epochs-300_lr-0.001'
        params, VAEmodel = load_VAE_model('./models/models', hyperparam_str)
        gen_path = os.path.join('./generated_samples', hyperparam_str)
        if gen_data:
            print('Generating node dataset..')
            node_data = generate_objects_VAE(params, VAEmodel, num_graphs)
            os.makedirs(gen_path, exist_ok=True)
            pickle.dump(node_data, open(os.path.join(gen_path, 'generated_objects.p'), 'wb') )
        else:
            print('Loading node dataset..')
            node_data = pickle.load(open(os.path.join(gen_path, 'generated_objects.p'), 'rb'))
        # generate edges
        hyperparam_str = 'BL1_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710'
        params, config, models = load_models('./models/models', hyperparam_str)
        gen_path = os.path.join('./generated_samples/', 'beta1_lsize16_1sample_without_coupling_batch-64_epochs-300_lr-0.001_'+hyperparam_str)
        if gen_data:
            print('Generating edge dataset..')
            graphs_data = generate_scene_graphs_given_nodes(gen_path, params, config, node_data, models, 1, ind_to_classes, ind_to_predicates, make_visuals=True)
        else:
            print('Loading edge dataset..')
            graphs_data = pickle.load(open(os.path.join(gen_path, 'generated_objects.p'), 'rb'))
        node_data = get_node_set(graphs_data)
    else:
        print('Using graphRNN model')
        # generate graphs
        #hyperparam_str = 'BL2_ordering-predefined_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPTrue_batch-128_samples-8_epochs-400_nlr-0.001_nlrdec-0.95_nstep-71_elr-0.001_elrdec-0.95_estep-71'
        hyperparam_str = 'new_sggen_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710'
        params, config, models = load_models('./models/models', hyperparam_str)
        ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
        gen_path = os.path.join('./generated_samples/', hyperparam_str)
        if gen_data:
            print('Generating graph dataset..')
            class_dict, _ = prior_distribution(graphs_train, ordering)
            graphs_data = generate_scene_graphs(gen_path, params, config, models, num_graphs, class_dict, ind_to_classes, ind_to_predicates, make_visuals=True)
            pickle.dump(graphs_data, open(os.path.join(gen_path, 'generated_objects.p'), 'wb') )
        else:
            print('Loading graph dataset..')
            graphs_data = pickle.load(open(os.path.join(gen_path, 'generated_objects.p'), 'rb'))
        node_data = get_node_set(graphs_data)

    print('MMD evaluation of nodes..')
    test_node_data = get_node_set(graphs_test)
    shuffle(test_node_data)
    shuffle(node_data)    
    t1 = time.time()
    test_gen_mmd = compute_mmd_node(node_data[:num_eval], test_node_data[:num_eval])
    t2 = time.time()
    print('MMD b/w test-generated on node set kernel: ', test_gen_mmd, '. Time taken: ', t2-t1)

    print('MMD evaluation of graphs..')
    shuffle(graphs_test)
    shuffle(graphs_data)
    t1 = time.time()
    test_gen_mmd = compute_mmd_graph(graphs_data[:num_eval], graphs_test[:num_eval])
    t2 = time.time()
    print('MMD b/w test-generated on graph kernel: ', test_gen_mmd, '. Time taken: ', t2-t1)
    