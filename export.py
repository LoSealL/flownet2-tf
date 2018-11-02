import argparse
import tensorflow as tf
from src.flownet2.flownet2 import FlowNet2
from src.flownet_s.flownet_s import FlowNetS
from src.flownet_sd.flownet_sd import FlowNetSD
from src.flownet_c.flownet_c import FlowNetC
from src.flownet_cs.flownet_cs import FlowNetCS
from src.flownet_css.flownet_css import FlowNetCSS

parser = argparse.ArgumentParser()
parser.add_argument("name", help="model name to export",
                    choices=('flownet2', 'flownets', 'flownetsd',
                             'flownetc', 'flownetcs', 'flownetcss'))
parser.add_argument("--dir", default='./exported', help="export dir")


def main():
    flag = parser.parse_args()
    if flag.name.lower() == 'flownet2':
        net = FlowNet2(2)
        ckpt = 'checkpoints/FlowNet2/flownet-2.ckpt-0'
    elif flag.name.lower() == 'flownets':
        net = FlowNetS(2)
        ckpt = 'checkpoints/FlowNetS/flownet-S.ckpt-0'
    elif flag.name.lower() == 'flownetsd':
        net = FlowNetSD(2)
        ckpt = 'checkpoints/FlowNetSD/flownet-SD.ckpt-0'
    elif flag.name.lower() == 'flownetc':
        net = FlowNetC(2)
        ckpt = 'checkpoints/FlowNetC/flownet-C.ckpt-0'
    elif flag.name.lower() == 'flownetcs':
        net = FlowNetCS(2)
        ckpt = 'checkpoints/FlowNetCS/flownet-CS.ckpt-0'
    elif flag.name.lower() == 'flownetcss':
        net = FlowNetCSS(2)
        ckpt = 'checkpoints/FlowNetCSS/flownet-CSS.ckpt-0'
    else:
        return
    with tf.Session() as sess:
        net.restore(sess, ckpt)
        tf.logging.info("exporting to {}/{}".format(flag.dir, flag.name))
        builder = tf.saved_model.builder.SavedModelBuilder(flag.dir + '/' + flag.name)
        # build the signature_def_map
        input_a = sess.graph.get_tensor_by_name('flow/inputa:0')
        input_b = sess.graph.get_tensor_by_name('flow/inputb:0')
        output = net.flow_results
        inputs = {
            'input_a': tf.saved_model.utils.build_tensor_info(input_a),
            'input_b': tf.saved_model.utils.build_tensor_info(input_b)
        }
        outputs = {
            'output_flow': tf.saved_model.utils.build_tensor_info(output)
        }
        sig = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs, outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: sig
            },
            strip_default_attrs=True)
        builder.save()


if __name__ == '__main__':
    main()
