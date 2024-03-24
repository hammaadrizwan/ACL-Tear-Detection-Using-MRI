if __name__ == "__main__":
    args = parse_arguments()
    run(args)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, choices=[0, 1], default=5)

    args = parser.parse_args()
    return args

def run(args):
    log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    # data augmentation pipeline
    augmentor = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    train_dataset = MRDataset('./data/', args.task, args.plane, transform=augmentor, train=True)
    validation_dataset = MRDataset('./data/', args.task, args.plane, train=False)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

    validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)

    mrnet = model.MRNet()
    mrnet = mrnet.cuda()

    optimizer = optim.Adam(mrnet.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience


    for epoch in range(num_epochs):

        train_loss, train_auc = train_model(
            mrnet, train_loader, epoch, num_epochs, optimizer, writer)
        val_loss, val_auc = evaluate_model(
            mrnet, validation_loader, epoch, num_epochs, writer)

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3}".format(
            train_loss, train_auc, val_loss, val_auc))

        if args.lr_scheduler == 1:
            scheduler.step(val_loss)

        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'model_{args.task}_{args.plane}_val_auc_\
                            {val_auc:0.4f}_train_auc_{train_auc:0.4f}\
                            _epoch_{epoch+1}.pth'
                for f in os.listdir('./models/'):
                    if (args.task in f) and (args.plane in f):
                        os.remove(f'./models/{f}')
                torch.save(mrnet, f'./models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, log_every=100):
    _ = model.train()
    if torch.cuda.is_available():
        model.cuda()
    y_preds = []
    y_trues = []
    losses = []

    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        prediction = model.forward(image.float())
        
        loss = F.binary_cross_entropy_with_logits(
            prediction[0], label[0], weight=weight[0])

        loss.backward()
        optimizer.step()
        
        y_pred = torch.sigmoid(prediction).item()
        y_true = int(label.item())

        y_preds.append(y_pred)
        y_trues.append(y_true)

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        loss_value = loss.item()
        losses.append(loss_value)

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]\
                    | avg train loss {4} | train auc : {5}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4)
                  )
                  )
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def extract_predictions(task, plane, train=True):
    assert task in ['acl', 'meniscus', 'abnormal']
    assert plane in ['axial', 'coronal', 'sagittal']
    
    models = os.listdir('../models/')

    model_name = list(filter(lambda name: task in name and plane in name, models))[0]
    model_path = f'../models/{model_name}'

    mrnet = torch.load(model_path)
    _ = mrnet.eval()
    
    train_dataset = MRDataset('../data/', 
                              task, 
                              plane, 
                              transform=None, 
                              train=train, 
                              normalize=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=1, 
                                               shuffle=False, 
                                               num_workers=10, 
                                               drop_last=False)
    predictions = []
    labels = []
    with torch.no_grad():
        for image, label, _ in tqdm_notebook(train_loader):
            logit = mrnet(image.cuda())
            prediction = torch.sigmoid(logit)
            predictions.append(prediction.item())
            labels.append(label.item())

    return predictions, labels


task = 'acl'
results = {}

for plane in ['axial', 'coronal', 'sagittal']:
    predictions, labels = extract_predictions(task, plane)
    results['labels'] = labels
    results[plane] = predictions
    
X = np.zeros((len(predictions), 3))
X[:, 0] = results['axial']
X[:, 1] = results['coronal']
X[:, 2] = results['sagittal']

y = np.array(labels)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X, y)

task = 'acl'
results_val = {}

for plane in ['axial', 'coronal', 'sagittal']:
    predictions, labels = extract_predictions(task, plane, train=False)
    results_val['labels'] = labels
    results_val[plane] = predictions

y_pred = logreg.predict_proba(X_val)[:, 1]
metrics.roc_auc_score(y_val, y_pred)