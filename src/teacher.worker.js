const tf = require('@tensorflow/tfjs')

export function teach(
  randomPointsArray,
  nu,
  weights01Array,
  weights02Array,
  weights2Array,
  teachOutputArray,
  count
) {
  tf.setBackend('cpu')

  let randomPoints = tf.tensor(randomPointsArray)
  let weights01 = tf.tensor(weights01Array)
  let weights02 = tf.tensor(weights02Array)
  let weights2 = tf.tensor(weights2Array)
  let teachOutput = tf.tensor(teachOutputArray)

  const randomSliceIndex = tensor => {
    return (0 + Math.random() * (tensor.shape[0] - 1 - 0)) | 0
  }

  let input = tf.concat([randomPoints, tf.onesLike(teachOutput)], 1)
  let e6
  let y6

  for (let i = 0; i < count; i++) {
    //Прямой прогон
    //Считаем выходы слоев
    console.log(i)
    let randomIndex = randomSliceIndex(input)
    let slicedInput = input.slice(randomIndex, 1)
    let slicedOutput = teachOutput.slice(randomIndex, 1)

    let net4 = tf.dot(slicedInput, weights01)
    let net5 = tf.dot(slicedInput, weights02)

    let y4 = net4.sigmoid()
    let y5 = net5.sigmoid()

    let hiddenOutput = tf.concat([tf.onesLike(y4), y4, y5], 1)

    let net6 = tf.dot(hiddenOutput, weights2)
    y6 = net6.sigmoid()

    //Считаем ошибки
    e6 = slicedOutput
      .sub(y6)
      .mul(y6)
      .mul(tf.onesLike(y6).sub(y6))

    let e5 = y5
      .mul(tf.onesLike(y5).sub(y5))
      .mul(e6)
      .mul(weights2.slice(2, 1))

    let e4 = y4
      .mul(tf.onesLike(y4).sub(y4))
      .mul(e6)
      .mul(weights2.slice(1, 1))

    // Считаем дельты весов
    let dW65 = y5.mul(e6).mul(tf.scalar(nu))
    let dW64 = y4.mul(e6).mul(tf.scalar(nu))
    let dW63 = tf
      .onesLike(y4)
      .mul(e6)
      .mul(tf.scalar(nu))

    let dW2 = tf.concat([dW65, dW64, dW63], 1)

    let dW52 = e5.mul(slicedInput.transpose().slice(0, 1)).mul(tf.scalar(nu))

    let dW51 = e5.mul(slicedInput.transpose().slice(1, 1)).mul(tf.scalar(nu))

    let dW50 = e5.mul(tf.scalar(1)).mul(tf.scalar(nu))

    let dW02 = tf.concat([dW52, dW51, dW50], 1)

    let dW42 = e4.mul(slicedInput.transpose().slice(0, 1)).mul(tf.scalar(nu))

    let dW41 = e4.mul(slicedInput.transpose().slice(1, 1)).mul(tf.scalar(nu))

    let dW40 = e4.mul(tf.scalar(1)).mul(tf.scalar(nu))

    let dW01 = tf.concat([dW42, dW41, dW40], 1)

    // Корректировка весов
    weights2 = weights2.add(dW2.transpose())
    weights02 = weights02.add(dW02.transpose())
    weights01 = weights01.add(dW01.transpose())
  }

  console.log('Веса после тренировки:')
  console.log('входной слой')
  weights01.print()

  console.log('скрытый слой')
  weights02.print()

  console.log('выходной слой')
  weights2.print()

  console.log('Ошибка:')
  e6.abs()
    .mean()
    .print()

  return {
    weights01: weights01.arraySync(),
    weights02: weights02.arraySync(),
    weights2: weights2.arraySync(),
  }
}
