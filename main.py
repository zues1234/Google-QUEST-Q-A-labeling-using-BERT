def training(index):
    TRAIN_BS = 16
    MAX_LEN = 512
    TEST_BS = 8
    EPOCHS = 20
    DEVICE = xm.xla_device()

    df = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")
    train_df, valid_df = model_selection.train_test_split(df, test_size=0.1, random_state=45)
    
    train_df.reset_index(drop=True)
    valid_df.reset_index(drop=True)

    sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
    columns = list(sample.drop("qa_id", axis=1).columns)

    train_targets = train_df[columns].values
    valid_targets = valid_df[columns].values

    tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-uncased")

    train_dataset = BERTdataset(
        qtitle = train_df.question_title.values,
        qbody = train_df.question_body.values,
        answer = train_df.answer.values,
        targets = train_targets,
        tokenizer= tokenizer,
        max_len= MAX_LEN
    )

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas = xm.xrt_world_size(), #gets the number of devices
        rank = xm.get_ordinal(),
        shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = TRAIN_BS,
        sampler = train_sampler, #replace shuffle with sampler for multiprocessing on TPU
    )

    valid_dataset = BERTdataset(
        qtitle = valid_df.question_title.values,
        qbody = valid_df.question_body.values,
        answer = valid_df.answer.values,
        targets = valid_targets,
        tokenizer= tokenizer,
        max_len= MAX_LEN
    )

    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset,
        num_replicas = xm.xrt_world_size(),    #gets the number of devices
        rank = xm.get_ordinal(),
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = TEST_BS,
        sampler= valid_sampler    #replace shuffle with sampler for multiprocessing on TPU
    )

    model = BERTModel("../input/bert-base-uncased")
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr = (3e-5 * xm.xrt_world_size())) 
    num_training_steps = int((len(train_dataset)/TRAIN_BS/xm.xrt_world_size()) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = num_training_steps)
    
    for epoch in range(EPOCHS):
        train_pl = pl.ParallelLoader(train_dataloader, [DEVICE])
        valid_pl = pl.ParallelLoader(valid_dataloader, [DEVICE])

        train_fn(model,train_pl.per_device_loader(DEVICE),optimizer,device=DEVICE,scheduler=scheduler)
        
        output, target = eval_fn(model, valid_pl.per_device_loader(DEVICE),device=DEVICE)

        spear=[]
        for i in range(target.shape[1]):
            p1 = list(target[:,i])
            p2 = list(output[:,i])
            coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            spear.append(coef)
        spear = np.mean(spear)
        xm.master_print(f"epoch = {epoch}, spearman rank = {spear}")
        xm.save(model.state_dict(), "./model.bin")



if __name__ == "__main__":
    xmp.spawn(training, nprocs=8, start_method='fork')