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
var LABELS = [];
var INPUTS = [];
var WEIGHTS = null;
var DATA_SIZE = 10; // The data set is 10,000 we are using only 10 examples here!

function preload() {
  console.log("👉 Preload");
  RAW_DATA = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");
  prepareData();
  trainModel();
}

function prepareData() {
  /* 
  From the data.csv

  Each row is like this:-

  1,255,255,255,0,0,0,0,0,0


  1 is the label
  255,255,255,0,0,0,0,0,0 is the input data
  */

  console.log("👉 prepareData");
  let ys = []; // [...]
  let xs = []; // [ [...],[...]]
  for (let row of RAW_DATA.getArray().slice(0, DATA_SIZE)) {
    row = row.map(x => x.trim()).map(x => parseInt(x));
    ys.push(row[0]);
    xs.push(row.slice(1));
  }
  console.log(ys);
  console.log(xs);

  INPUTS = tf.tensor2d(xs);
  LABELS = tf.tensor1d(ys);

  INPUTS.print();
  LABELS.print();
}

function trainModel() {
  console.log("👉 Train");

  // Initialise the Weights
  WEIGHTS = tf.variable(tf.randomNormal([1, 9]));
  console.log("WEIGHTS ");
  WEIGHTS.print();

  function predict(inputs) {
    // Given an input tensor like
    // [[255,255,255,0,0,0,0,0,0],...]
    // Will return an output tensor which matches the labels like
    // [1,..]
    return inputs
      .mul(WEIGHTS)
      .sum((axis = 1))
      .step(0);
  }

  function loss(predicted, actual) {
    // The loss between the actual labels tensor [1,0,1,0,1...] and what the model predicted [1,0,0,0,0...]
    // Needs to return a number which if lower means we have matched more rows
    // I'm taking one away from the other, suming the difference and then squaring.

    let x = predicted // So e.g. [1,1,1] - [1,0,0]
      .sub(actual) // should result in [0,1,1]
      .sum() // should result in 2
      .square(); // should result in 4
    return x;
  }

  console.log("INPUTS ");
  INPUTS.print();
  console.log("LABELS ");
  LABELS.print();
  console.log("PREDICT ");
  predict(INPUTS).print();
  console.log("COST FUNCTION ");
  loss(predict(INPUTS), LABELS).print();

  const optimizer = tf.train.sgd(0.1);

  for (let i = 0; i < 10; i++) {
    tf.tidy(() => {
      const returnCost = true;
      let cost = optimizer.minimize(() => {
        let PREDICTIONS = predict(INPUTS);
        return loss(PREDICTIONS, LABELS);
      }, returnCost);
      console.log(`LOSS [${i}]: ${cost.dataSync()}`);
      console.log(`EPOC WEIGHTS: ${WEIGHTS.dataSync()}`);
    });
  }
}
