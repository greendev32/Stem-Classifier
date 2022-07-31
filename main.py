# Stem Classifier Main
# Entry point for the stem classifier model

from stemClassifier import stemClassifier

if __name__ == '__main__':
    sc = stemClassifier.StemClassifier()

    # Original dataset
    #sc.loadDataset('C:/Users/green/Pictures/DATA/split_dataset/')
    #sc.createModel()
    #sc.trainModel('my_model')
    #sc.classify('C:/Users/green/PycharmProjects/stemClassifier/my_model', 'C:/Users/green/Pictures/DATA/test/')

    # Simplified dataset
    # sc.loadDataset('C:/Users/green/Pictures/DATA/simplified_roi_dataset/shuffled/')
    # sc.createModel()
    # sc.trainModel('simplified_model')
    #sc.classify('C:/Users/green/PycharmProjects/stemClassifier/simplified_model', 'C:/Users/green/Pictures/DATA/simplified_roi_dataset/test_set/')

    # ResNet50 Test
    sc.loadDataset('C:/Users/green/Pictures/DATA/split_dataset/')
    sc.createResNet50Model()
    sc.trainModel('resnet50_model')
    #sc.classify('C:/Users/green/PycharmProjects/stemClassifier/resnet50_model',
    #            'C:/Users/green/Pictures/DATA/test/')
