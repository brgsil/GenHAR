
from sklearn.metrics import accuracy_score, pairwise_distances
from scipy.stats import wasserstein_distance

class Evaluator:
    def __init__(self,config,df_train,df_test,df_val,df_synthetic):
        print(config["evaluations"])
        self.df_train=df_train
        self.df_test=df_test
        self.df_val=df_val
        self.df_synthetic=df_synthetic
        self.config=config
        self.dataset=config['dataset']
        self.transform=config['transform']
        self.model=config['model']

    def get_visualization(self):
        print("visual")
        tsne=False
        tsne_glob=True
        bar_count_p=False
        images_p=False
        X_train = self.df_train.drop(columns=['label']).values
        y=self.df_train['label'].values
        X_gen=self.df_synthetic.drop(columns=['label']).values
        y_gen=self.df_synthetic['label'].values
        label_names = ['sit', 'stand', 'walk', 'stair up', 'stair down', 'run']
        if tsne:
            import eval.visualization.scaterplotly as sp
            

            #sp.get_plotly_by_labels(X_train,X_gen,y,y_gen)
            import eval.visualization.scaterplotly as sp

            
            sp.tsne_subplots_by_labels(X_train, y, X_gen, y_gen, label_names=label_names)

        if bar_count_p:
            from eval.visualization import bar_count
            #bar_count.plotly_count_by_labels(X_gen, y_gen, class_names=label_names)
            bar_count.plotly_count_by_labels_compare(X_train, y, X_gen, y_gen, class_names=None)

        if images_p:
             from eval.visualization import visualize_images
             label_name = 1  # or None to plot all samples
             #visualize_images.plot_sample_comparison(X_train, y, X_gen, y_gen, label=label_name, n_samples=1, reshape=True)
             #visualize_images.visualize_original_and_reconst_ts(X_train[0:limit], X_gen, num=10)




        if tsne_glob:
            from eval.visualization import visualization_tsne_r_s
            filename=f'reports/{self.dataset}_{self.transform}_{self.model}.pdf' 
            visualization_tsne_r_s.visualize_tsne_r_s(self.df_train,self.df_synthetic,path=filename)

    def get_metrics(self):
                print("visual")
    
                
    def get_ml(self):
          print("mch")

    
    
    def fidelity(real_data, synthetic_data):
        return wasserstein_distance(real_data.ravel(), synthetic_data.ravel())

    
    
    def utility(real_data, synthetic_data, classifier):
        classifier.fit(synthetic_data, real_data)
        predictions = classifier.predict(real_data)
        return accuracy_score(real_data, predictions)

   
   
    def diversity(synthetic_data):
        return pairwise_distances(synthetic_data).mean()