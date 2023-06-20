#!/usr/bin/env node
const fs = require('fs');
const argparse = require('argparse');

function parseArgs() {
    const parser = new argparse.ArgumentParser({
        description:
            'TensorFlow.js: Evaluating an efficientnet-b2 Model',
        addHelp: true
    });
    parser.addArgument('dataset', {
        type: 'string',
        help: 'Path of the dataset.'
    });
    parser.addArgument('modelSavePath', {
        type: 'string',
        help: 'Path at which the model to be evaluated is saved.'
    });
    // parser.addArgument('--batchSize', {
    //     type: 'int',
    //     defaultValue: 1,
    //     help: 'Batch size to be used during model training.'
    // });
    parser.addArgument('--gpu', {
        action: 'storeTrue',
        help: 'Use tfjs-node-gpu for evaluation (requires CUDA-enabled ' +
            'GPU and supporting drivers and libraries.'
    });
    parser.addArgument('--normalize', {
        type: 'string',
        choices:['ImagenetCenteredRgb', 'ImagenetMeanStd'],
        defaultValue: 'ImagenetMeanStd',
        help: ''
    });
    return parser.parseArgs();
}

function normalize(img, mean, std, axis) {
    const centeredRgb = new Array(img.shape[axis]).fill(0).map((idx) =>
        tf.gather(img, [idx], axis)
            .sub(tf.scalar(mean[idx]))
            .div(tf.scalar(std[idx])));

    return tf.concat(centeredRgb, axis);
}

async function main() {
    const args = parseArgs();
    if (args.gpu) {
        tf = require('@tensorflow/tfjs-node-gpu');
    } else {
        tf = require('@tensorflow/tfjs-node');
    }

    console.log(`Loading model from ${args.modelSavePath}...`);
    const model = await tf.loadGraphModel(`file://${args.modelSavePath}`);

    const csvDataset = tf.data.csv(`file://${args.dataset}/gt.csv`, {
        columnConfigs: {
            target: {
                isLabel: true
            }
        }
    });

    const flattenedDataset =
        csvDataset
            .map(({ xs, ys }) => {
                const imageBuffer = fs.readFileSync(`${args.dataset}/images/${xs['image']}`);
                let tensor;
                if (args.normalize == 'ImagenetMeanStd') {
                    tensor = tf.node.decodeImage(imageBuffer, 3)
                        .div(255.0)
                        .resizeNearestNeighbor([224, 224])
                        .transpose([2, 0, 1])
                        ;
                    mean = [0.485, 0.456, 0.406];
                    std = [0.229, 0.224, 0.225];
                    tensor = tensor.sub(tf.reshape(mean, [mean.length, 1, 1]))
                    tensor = tensor.div(tf.reshape(std, [std.length, 1, 1]));

                } else if (args.normalize == 'ImagenetCenteredRgb'){
                    tensor = tf.node.decodeImage(imageBuffer, 3)
                        .div(255.0)
                        .resizeNearestNeighbor([224, 224])
                        .expandDims()
                        .transpose([0, 3, 1, 2])
                        ;
                    tensor = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1)
                    tensor = tf.squeeze(tensor);
                }else{
                    throw new Error('NotImplemented');
                }

                return { xs: tensor, ys: ys }
            }).batch(1);

    const iterator = await flattenedDataset.iterator();
    predictions = [];
    targets = [];

    while (true) {
        const item = await iterator.next();
        if (item.done) {
            break;
        }
        const prediction = await model.predictAsync(item.value.xs, { batch_size: 1 });

        output = tf.argMax(prediction, axis = 1);
        target = item.value.ys['target'];

        predictions.push(output.dataSync()[0]);
        targets.push(target.dataSync()[0])
    }

    // console.log(predictions.length);
    // console.log(targets.length);
    console.log(tf.mean(tf.equal(predictions, targets)).dataSync()[0].toFixed(6));
}

if (require.main === module) {
    main();
}
