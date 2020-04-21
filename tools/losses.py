import keras
import keras.backend as K 

def jensen_shannon_divergence(y_true, y_pred):
    """Computes Jensen-Shannon divergence loss between `y_true` and `y_pred`.
    
    Parameters
    ----------
    y_true : Tensor
        Tensor of true targets
    y_pred : Tensor
        Tensor of predicted targets
    
    Returns
    -------
    Tensor
        A `Tensor` with loss.

    Usage
    -----
    ```python    
    loss = custom.losses.jensen_shannon_divergence([.4, .9, .2], [.5, .8, .12])
    print('Loss: ', loss.numpy())  # Loss: 0.11891246
    ```
    """

    y_mean = 0.5 * y_true + 0.5 * y_pred
    jsd = 0.5 * keras.losses.kullback_leibler_divergence(y_true, y_mean) \
         + 0.5 * keras.losses.kullback_leibler_divergence(y_pred, y_mean)
    return jsd / K.log(2.0)