/***
 *
 *
 * We want to train a model which will return 1 if the bottom 3 numbers are greater than the top three numbers, else 0
 *
 * e.g.
 *
 * 255,255,255
 *   0,  0,  0
 *   0,  0,  0
 *
 * Should return 0 and
 *
 *   0,  0,  0
 *   0,  0,  0
 * 255,255,255
 *
 * Should return 1
 *
 * We can train a simple model to solve this by multiplying the input matrix by another with 9 weights
 *
 *  -1, -1, -1     255, 255, 255
 *   0,  0,  0  x    0,   0,   0
 *   1,  1,  1       0,   0,   0
 *
 * Then if we sum up all the numbers the result should be -ve if the top row is higher or +ve if the bottom row is higher.
 *
 * The goal is to train a model using the input data which results in 9 weights like the above.
 *
 */

var RAW_DATA = null;
var WEIGHTS = null;
var EPOCHS = 100;
var BATCHER = null;
var BATCH_SIZE = 2000;

// tf.ENV.set("DEBUG", true);

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
  let labels = []; // [...]
  let inputs = []; // [ [...],[...]]
  for (let row of RAW_DATA.getArray()) {
    row = row.map(x => x.trim()).map(x => parseInt(x));
    labels.push(row[0] * 2 - 1); // Conver to -1 and 1
    inputs.push(row.slice(1).map(x => x / 255));
  }
  BATCHER = new Batcher(labels, inputs, BATCH_SIZE);
}

function createWeights() {
  // Initialise the Weights
  console.log("👉 createWeights");

  // This should be the IDEAL set of target weights!
  WEIGHTS = tf.variable(
    tf.tensor([
      [1 / 3.0, 1 / 3.0, 1 / 3.0, 0, 0, 0, -1 / 3.0, -1 / 3.0, -1 / 3.0]
    ])
  );

  // WEIGHTS = tf.variable(tf.truncatedNormal([1, 9]), true);

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
  return inputs.matMul(WEIGHTS.transpose());
  // .step(0);
}

function loss(predicted, actual) {
  let x = predicted // So e.g. [1,1,1] - [1,0,0]
    .sub(actual) // should result in [0,1,1]
    .square()
    .mean();

  return x;
}

function validateModel() {
  console.log("👉 validateModel");

  // This should be the IDEAL set of target weights!
  WEIGHTS = tf.variable(
    tf.tensor([
      [1 / 3.0, 1 / 3.0, 1 / 3.0, 0, 0, 0, -1 / 3.0, -1 / 3.0, -1 / 3.0]
    ])
  );

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
}

function trainModel() {
  const optimizer = tf.train.sgd(0.001);

  // We need to run for each batch and each epoch
  const trainingSteps = EPOCHS * BATCHER.batchCount;

  (async () => {
    for (let i = 0; i < trainingSteps; i++) {
      tf.tidy(() => {
        const returnCost = true;

        // Calculate how many batches
        const batchNum = i % BATCHER.batchCount;
        let cost = optimizer.minimize(() => {
          // Get inputs and labels for this batch
          const { inputs, labels } = BATCHER.nextBatch(batchNum);

          // Calculate loss
          return loss(predict(inputs), labels.transpose());
        }, returnCost);

        if (i % 100 === 0) {
          console.log(`LOSS [${i}]: ${cost.dataSync()}`);
          console.log(`EPOC WEIGHTS: `);
          WEIGHTS.print();
        }
      });
      await tf.nextFrame();
    }
  })();
}
