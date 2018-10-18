class Batcher {
  constructor(labels, inputs, batchSize) {
    this.labels = labels;
    this.inputs = inputs;
    this.totalSize = this.inputs.length;
    this.batchSize = batchSize;
    this.batchCount = parseInt(Math.ceil(this.totalSize / this.batchSize));
  }

  nextBatch(batchNum) {
    let startCount = batchNum * this.batchSize;
    let endCount = startCount + this.batchSize;
    if (endCount > this.totalSize) {
      throw new Error(
        `Requesting more data than we have, ${startCount}:${endCount}:${
          this.totalSize
        }`
      );
    }

    return {
      labels: tf.tensor(this.labels.slice(startCount, endCount)),
      inputs: tf.tensor(this.inputs.slice(startCount, endCount))
    };
  }
}
