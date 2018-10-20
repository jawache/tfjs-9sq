// Configuration
var EPOCHS = 2000;

// Variables
var RAW_DATA = null;
var WEIGHTS = null;
var DATA = null;
var PERFORMANCE = null;
var CANVAS = null;
var DRAW_CANVAS = false;

function preload() {
  console.log("👉 Preload");
  RAW_DATA = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");

  prepareData();

  setupCanvas();

  validateModel();
  createWeights();
  trainModel();
}

function setupCanvas() {
  frameRate(5);
  createCanvas(windowWidth, windowHeight);

  CANVAS = new Panel("9Squares by Asim Hussain", 0, 0);

  const dashboardPanel = new Panel("Testing Data", 0, 0);
  dashboardPanel.add(new Squares(DATA));

  const detailsPanel = new Panel("Details Data", 450, 0);
  detailsPanel.add(new Details(DATA));

  CANVAS.add(dashboardPanel).add(detailsPanel);
}

function draw() {
  background(50);

  translate(10, 10);
  fill(255)
    .strokeWeight(0)
    .textSize(16)
    .textFont("Helvetica", 24);
  text("9 Squares", 0, 24);

  CANVAS.draw();
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

  const inputs = tf.tensor(DATA.training.inputs);
  const labels = tf.tensor([DATA.training.labels]).transpose(); // We need to convert into columns

  const testing_inputs = tf.tensor(DATA.testing.inputs);
  const testing_labels = tf.tensor([DATA.testing.labels]).transpose(); // We need to convert into columns

  for (let i = 0; i <= EPOCHS; i++) {
    tf.tidy(() => {
      let cost = optimizer.minimize(() => {
        return loss(predict(inputs), labels);
      }, true);
      // console.log(`[${i}] ${cost.dataSync()[0]}`);

      if (i % 10 === 0) {
        // Calculate accuracy
        console.log(`[${i}]======================================`);

        console.log(`LOSS:`);
        cost.print();
        console.log(`EPOC WEIGHTS: `);
        WEIGHTS.print();
        console.log("-- TESTING --");
        const predictions = predict(testing_inputs);
        console.log("LOSS: ");
        const testingLoss = loss(predictions, testing_labels);
        testingLoss.print();
        console.log("ACCURACY: ");
        const testingAcc = acc(predictions, testing_labels);
        testingAcc.print();

        DATA.testing.loss = cost.dataSync()[0];
        DATA.testing.weights = WEIGHTS.dataSync();
        DATA.testing.predictions = predictions.sign().dataSync();
        DATA.testing.testingLoss = testingLoss.dataSync()[0];
        DATA.testing.testingAcc = testingAcc.dataSync()[0];
        DATA.testing.epoch = i;
      }
    });
    await tf.nextFrame();
  }
}
