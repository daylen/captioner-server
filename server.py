from captioner import Captioner

weights_path = './lrcn_finetune_vgg_trainval_iter_100000.caffemodel'
image_net_proto = './VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = './lrcn_word_to_preds.deploy.prototxt'
vocab_path = './vocabulary.txt'

c = Captioner(weights_path, image_net_proto, lstm_net_proto, vocab_path)

def get_caption(fname):
	descriptor = c.image_to_descriptor(fname)
	indices = c.predict_caption(descriptor)[0][0]
	return c.sentence(indices)