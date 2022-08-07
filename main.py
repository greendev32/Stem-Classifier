# Stem Classifier Main
# Entry point for the stem classifier model

from stemClassifier import stemClassifier

if __name__ == '__main__':
    sc = stemClassifier.StemClassifier()

    # Original dataset
    # sc.loadDataset('C:/Users/green/Pictures/DATA/split_dataset/',
    #                200,
    #                200)
    #sc.createModel()
    #sc.trainModel('my_model')
    # sc.classify('C:/Users/green/PycharmProjects/stemClassifier/my_model',
    #             'C:/Users/green/Pictures/DATA/test/',
    #             200,
    #             200)
    # sc.classifyROI('C:/Users/green/PycharmProjects/stemClassifier/my_model',
    #                'C:/Users/green/Pictures/DATA/stem_dataset/',
    #                True)

    # Simplified dataset
    # sc.loadDataset('C:/Users/green/Pictures/DATA/simplified_roi_dataset/shuffled/',
    #                400,
    #                400)
    #sc.createModel()
    #sc.trainModel('quadrant')
    # sc.classify('C:/Users/green/PycharmProjects/stemClassifier/batch_normalization',
    #             'C:/Users/green/Pictures/DATA/simplified_roi_dataset/test_set/',
    #             400,
    #             400)
    # sc.classifyROI('C:/Users/green/PycharmProjects/stemClassifier/batch_normalization',
    #                'C:/Users/green/Pictures/DATA/simplified_dataset2/',
    #                False)

    # ResNet50 Test
    # sc.loadDataset('C:/Users/green/Pictures/DATA/simplified_roi_dataset/shuffled/',
    #                400,
    #                400)
    #sc.createResNet50Model()
    #sc.trainModel('resnet50_model')
    # sc.classify('C:/Users/green/PycharmProjects/stemClassifier/resnet50_model',
    #             'C:/Users/green/Pictures/DATA/simplified_roi_dataset/test_set/',
    #             400,
    #             400)
