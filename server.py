from flask import Flask, request, redirect, url_for, jsonify
from time import time
from captioner import Captioner
from scipy.misc import imread

weights_path = './lrcn_finetune_vgg_trainval_iter_100000.caffemodel'
image_net_proto = './VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = './lrcn_word_to_preds.deploy.prototxt'
vocab_path = './vocabulary.txt'

# 0 = GPU
c = Captioner(weights_path, image_net_proto, lstm_net_proto, vocab_path, 0)

def get_caption(fname):
	descriptor = c.image_to_descriptor(fname)
	indices = c.predict_caption(descriptor)[0][0]
	return c.sentence(indices)

app = Flask(__name__)

@app.route('/')
def index():
	return 'Image Captioning as a Service. Usage: /upload'

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            start = time()
            img = imread(file)
            print img.shape
            caption = get_caption(img)
            print caption
            print 'finished nn', time() - start
            json = {'caption': caption, 'time': time() - start}
            return jsonify(json)
        else:
            return '''
            <!doctype html>
            <h1>Error</h1>
            <p>Please upload a JPEG file.</p>
            '''
    return '''
    <!doctype html>
    <h1>Upload picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)