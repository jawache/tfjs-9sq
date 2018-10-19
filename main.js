// Configuration
var EPOCHS = 3000;

// Variables
var RAW_DATA = null;
var WEIGHTS = null;
var DATA = null;

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
  DATA = new Data(RAW_DATA);
  console.log(DATA.training);
}

// Initialise the Weights
function createWeights() {
  console.log("👉 createWeights");
  // Create a weights tensor
  // This needs to be 9 rows and 1 column so in dot with the inputs it will generate 1 value
  WEIGHTS = tf.variable(tf.truncatedNormal([9, 1]), true);
  console.log("WEIGHTS -->");
  WEIGHTS.print();
}

function predict(inputs) {
  return inputs.dot(WEIGHTS);
}

function loss(predicted, actual) {
  return predicted // So e.g. [1,1,1] - [1,0,0]
    .sub(actual) // should result in [0,1,1]
    .square()
    .mean();
}

function acc(predicted, actual) {
  return tf
    .sign(predicted)
    .mul(actual)
    .mean();
}

function validateModel() {
  console.log("👉 validateModel");

  // This should be the IDEAL set of target weights!
  WEIGHTS = tf
    .variable(tf.tensor([[1, 1, 1, 0, 0, 0, -1, -1, -1]]))
    .transpose();

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

  const inputs = DATA.training.inputs;
  const labels = DATA.training.labels;

  for (let i = 0; i < EPOCHS; i++) {
    tf.tidy(() => {
      let cost = optimizer.minimize(() => {
        return loss(predict(inputs), labels);
      }, true);
      // console.log(`[${i}] ${cost.dataSync()[0]}`);

      if (i % 100 === 0) {
        const testingInputs = DATA.testing.inputs;
        const testingLabels = DATA.testing.labels;

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
