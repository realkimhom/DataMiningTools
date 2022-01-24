import numpy as np


def loss_function_linear(model, features, label, methods="USE"):
    """

    :param model:  [w0, w1, w2, ... , wn]
    :param features: should be a list of list. [[x11,.....,xn1], ...... ,[x1m, ....., xnm]]
    :param label: [y1,y2,....,yn]
    :param methods:
    :return:
    """
    assert len(model) - len(features[0]) == 1
    label_hat_list = []
    for feature in features:
        label_hat = model[0]
        for f in feature:
            for w in model[1:]:
                label_hat = label_hat + w * f
        label_hat_list.append(label_hat)
    label = np.array(label)
    if methods == "USE":
        print(label_hat_list - label)
        print(label)
        error = sum(np.power((label_hat_list - label), 2)) / len(features)
        return round(error, 4)


x = loss_function_linear([1, 0.5], [[1], [1.5], [3]], [1.6, 1.5, 2.4])
print(x)
