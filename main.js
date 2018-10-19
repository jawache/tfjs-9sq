var RAW_DATA = null;
var WEIGHTS = null;
var EPOCHS = 3000;
var BATCHER = null;
var BATCH_SIZE = 1000;

var TRAINING = {
  inputs: [],
  labels: []
};

var TESTING = {
  inputs: [],
  labels: []
};

// class Data {
//   constructor(trainingFeatures, testingFeatures) {
//     this.trainingFeatures = this.parse(trainingFeatures);
//     this.testingFeatures = this.parse(testingFeatures);
//   }
// }

function preload() {
  console.log("👉 Preload");
  RAW_DATA = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");
  prepareData();
  validateModel();
  createWeights();
  trainModel();
}

function prepareData() {
  console.log("👉 prepareData");

  // Use 80% as the training data
  const trainingData = RAW_DATA.getArray().slice(0, 2000);

  // Use 20% as the testing data
  const testingData = RAW_DATA.getArray().slice(2000, 2500);

  function parse(data, store) {
    store.labels = [];
    store.inputs = [];
    for (let row of data) {
      // Convert each item to an integer
      row = row.map(x => x.trim()).map(x => parseInt(x));

      // Get the first item, these are the labels
      store.labels.push(row[0]);

      // Get the remaining items in the row, divide by 255 to get from 0 -> 1
      store.inputs.push(row.slice(1).map(x => x / 255));
    }
    store.labels = [store.labels];
  }

  parse(trainingData, TRAINING);
  parse(testingData, TESTING);

  console.log(TRAINING);
}

function createWeights() {
  // Initialise the Weights
  console.log("👉 createWeights");

  // This should be the IDEAL set of target weights!
  // WEIGHTS = tf.variable(tf.tensor([[1, 1, 1, 0, 0, 0, -1, -1, -1]]));

  WEIGHTS = tf.variable(tf.truncatedNormal([1, 9]), true);

  console.log("WEIGHTS -->");
  WEIGHTS.print();
}

function predict(inputs) {
  // Given an input tensor like
  // [[255,255,255,0,0,0,0,0,0],...]
  // Will return an output tensor which matches the labels like
  // [1,..]
  // console.log("PREDICT ");
  // inputs.matMul(WEIGHTS.transpose()).print();
  return inputs.dot(WEIGHTS.transpose());
  // .step(0);
}

function loss(predicted, actual) {
  let x = predicted // So e.g. [1,1,1] - [1,0,0]
    .sub(actual) // should result in [0,1,1]
    .square()
    .mean();

  return x;
}

// function loss(predicted, actual) {
//   return tf
//     .sign(predicted)
//     .sub(actual)
//     .mean();
// }

function acc(predicted, actual) {
  return tf
    .sign(predicted)
    .mul(actual)
    .mean();
}

function validateModel() {
  console.log("👉 validateModel");

  // This should be the IDEAL set of target weights!
  // [1 / 3.0, 1 / 3.0, 1 / 3.0, 0, 0, 0, -1 / 3.0, -1 / 3.0, -1 / 3.0]
  WEIGHTS = tf.variable(tf.tensor([[1, 1, 1, 0, 0, 0, -1, -1, -1]]));

  // Small set of inputs
  const inputs = tf.tensor([
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0.5, 0, 0.3, 0, 0, 0.7, 0]
  ]);

  // Small set of labels, 1 means top 3 is higher, -1 means bottom 3 is higher
  const labels = tf.tensor([[1, -1, -1]]).transpose();

  console.log("PREDICTED LABELS -->");
  predict(inputs).print();
  console.log("ACTUAL LABELS -->");
  labels.print();
  console.log("LOSS -->");
  // For this to work I think the loss has to be 0 here, instead it's 0.2903703451156616
  loss(predict(inputs), labels).print();
  console.log("ACC -->");
  // For this to work I think the loss has to be 0 here, instead it's 0.2903703451156616
  acc(predict(inputs), labels).print();
}

async function trainModel() {
  const optimizer = tf.train.sgd(0.01);

  const inputs = tf.tensor(TRAINING.inputs);
  const labels = tf.tensor(TRAINING.labels).transpose();

  const testingInputs = tf.tensor(TESTING.inputs);
  const testingLabels = tf.tensor(TESTING.labels).transpose();

  for (let i = 0; i < EPOCHS; i++) {
    tf.tidy(() => {
      let cost = optimizer.minimize(() => {
        return loss(predict(inputs), labels);
      }, true);
      // console.log(`[${i}] ${cost.dataSync()[0]}`);

      if (i % 100 === 0) {
        // Calculate accuracy
        console.log(`[${i}]======================================`);
        console.log(`LOSS:`);
        cost.print();
        console.log(`EPOC WEIGHTS: `);
        WEIGHTS.print();
        console.log("-- TESTING --");
        const predictions = predict(testingInputs);
        console.log("LOSS: ");
        const testingLoss = loss(predictions, testingLabels);
        testingLoss.print();
        console.log("ACCURACY: ");
        const testingAcc = acc(predictions, testingLabels);
        testingAcc.print();
        console.log(`[${i}]======================================`);
      }
    });
    await tf.nextFrame();
  }
}
