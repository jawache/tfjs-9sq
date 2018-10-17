var RAW_DATA = null;
var LABELS = [];
var INPUTS = [];

function preload() {
  console.log("👉 Preload");
  data = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");
  prepareData();
  createModel();
  // trainModel();
}

function prepareData() {
  console.log("👉 prepareData");
  console.log(data);
  // Create an output and input tensors.
  let ys = []; // [...]
  let xs = []; // [ [...],[...]]
  for (let row of data.getArray().slice(0, 10)) {
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

function createModel() {
  let weights = tf.variable(tf.randomNormal([1, 9]));
  console.log("WEIGHTS ");
  weights.print();

  function predict(x) {
    return x.mul(weights).sum();
    // .step(0); // If x > 0 then 1 else 0
  }

  function loss(predicted, actual) {
    // Mean Squared Error
    // predicted.print();
    // actual.print();
    let x = predicted
      .sub(actual)
      .mean()
      .abs();
    // console.log("LOSS ");
    // x.print();
    LOSS = x.dataSync()[0];
    console.log("LOSS ", LOSS);
    return x;
    // return tf.scalar(1000);
  }

  console.log("INPUTS ");
  INPUTS.print();
  console.log("PREDICT ");
  predict(INPUTS).print();
  console.log("LOSS ");
  loss(predict(INPUTS), LABELS).print();

  // LABELS.print();

  weights.print();
  const optimizer = tf.train.sgd(0.5);
  for (let i = 0; i < 10; i++) {
    optimizer.minimize(() => {
      let predicted = predict(INPUTS);
      return loss(predicted, LABELS);
    });
    console.log("EPOC WEIGHTS ");
    weights.print();
  }

  // a.print();

  // When passed in the array of predictedYs calculates the mean square loss compared to the actualYs
  // function loss(predictedYs, actualYs) {
  //   // Mean Squared Error
  //   let x = predictedYs
  //     .sub(actualYs)
  //     .square()
  //     .mean();
  //   LOSS = x.dataSync()[0];
  //   return x;
  // }
}
