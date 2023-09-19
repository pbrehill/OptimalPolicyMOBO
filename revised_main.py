




ehvi_experiment = build_experiment()
ehvi_data = initialize_experiment(ehvi_experiment)

ehvi_hv_list = []
ehvi_model = None
for i in range(N_BATCH):
    ehvi_model = get_MOO_EHVI(
        experiment=ehvi_experiment,
        data=ehvi_data,
    )
    generator_run = ehvi_model.gen(1)
    trial = ehvi_experiment.new_trial(generator_run=generator_run)
    trial.run()
    ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])

    exp_df = exp_to_df(ehvi_experiment)
    outcomes = np.array(exp_df[['a', 'b']], dtype=np.double)
    try:
        hv = observed_hypervolume(modelbridge=ehvi_model)
    except:
        hv = 0
        print("Failed to compute hv")
    ehvi_hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[['a', 'b']], dtype=np.double)