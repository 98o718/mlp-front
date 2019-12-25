import React, { useEffect, useState } from 'react'
import {
  Card,
  CardText,
  CardBody,
  CustomInput,
  Col,
  Row,
  Form,
  FormGroup,
  Label,
  Button,
} from 'reactstrap'
import * as tf from '@tensorflow/tfjs'
// eslint-disable-next-line
import teacherWorker from 'workerize-loader!../../teacher.worker'
import InputNumber from 'rc-input-number'
import { TxtReader } from 'txt-reader'
import { saveAs } from 'file-saver'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'

const Generator = () => {
  const [k, setK] = useState(1)
  const [qty, setQty] = useState(10)
  const [sd1, setSd1] = useState(0.2)
  const [sd2, setSd2] = useState(0.3)
  const [count, setCount] = useState(1000)
  const [nu, setNu] = useState(0.5)
  const [b, setB] = useState(0)
  const [offset, setOffset] = useState(1)

  const M = 0
  const CLASS1 = 0.9
  const CLASS2 = 0.1

  const [randomPoints, setRandomPoints] = useState()
  const [scaled, setScaled] = useState()
  const [teachOutput, setTeachOutput] = useState()
  const [teached, setTeached] = useState(false)
  const [teaching, setTeaching] = useState(false)
  const [teacher, setTeacher] = useState()
  const [firstClass, setFirstClass] = useState()
  const [secondClass, setSecondClass] = useState()
  // const [line, setLine] = useState()

  const [file, setFile] = useState()
  const [isFile, setIsFile] = useState(false)

  const [weights01, setWeights01] = useState()
  const [weights02, setWeights02] = useState()
  const [weights2, setWeights2] = useState()

  tf.setBackend('cpu')
  tf.initializers.randomUniform({ seed: 1 })

  useEffect(() => {
    randomPoints && setScaled(tf.scalar(1).div(randomPoints))
    // eslint-disable-next-line
  }, [randomPoints])

  useEffect(() => {
    setTeacher(teacherWorker())

    resetWeights()

    generate()
    // eslint-disable-next-line
  }, [])

  useEffect(() => {
    generate()
    setTeached(false)
    // eslint-disable-next-line
  }, [qty, sd1, sd2, k, b, offset])

  useEffect(() => {
    resetWeights()
    // eslint-disable-next-line
  }, [k, b])

  useEffect(() => {
    if (isFile) {
      setFirstClass([])
      setSecondClass([])
    } else {
      resetWeights()
      generate()
    }
    // eslint-disable-next-line
  }, [isFile])

  const generate = () => {
    let x = i => 2 + i * 0.1
    let y = x => k * x + b

    let lineCoords = []
    let x1s = []
    let x2s = []

    for (let i = 0; i < qty; i++) {
      lineCoords.push({
        x: x(i),
        y: y(x(i)),
      })

      x1s.push(x(i))
      x2s.push(y(x(i)))
    }

    let firstClassDotX = tf
      .tensor([x1s])
      .add(tf.randomNormal([qty], M, sd1))
      .transpose()

    let firstClassDotY = firstClassDotX
      .mul(tf.scalar(k))
      .add(tf.scalar(b))
      .add(tf.scalar(offset))
      .add(tf.randomNormal([qty, 1], M, sd2))

    let firstClassTensor = tf.concat([firstClassDotX, firstClassDotY], 1)

    let secondClassDotX = tf
      .tensor([x1s])
      .add(tf.randomNormal([qty], M, sd1))
      .transpose()

    let secondClassDotY = secondClassDotX
      .mul(tf.scalar(k))
      .add(tf.scalar(b))
      .sub(tf.scalar(offset))
      .add(tf.randomNormal([qty, 1], M, sd2))

    let secondClassTensor = tf.concat([secondClassDotX, secondClassDotY], 1)

    let firstDots = firstClassTensor
      .arraySync()
      .map(point => ({ x: point[0], y: point[1] }))

    let secondDots = secondClassTensor
      .arraySync()
      .map(point => ({ x: point[0], y: point[1] }))

    setFirstClass(firstDots)
    setSecondClass(secondDots)

    let input = tf.concat([firstClassTensor, secondClassTensor])

    let output = tf.concat([
      tf.fill([qty, 1], CLASS1),
      tf.fill([qty, 1], CLASS2),
    ])
    setRandomPoints(input)
    setTeachOutput(output)

    // setLine(lineCoords)
  }

  const fetchWebWorker = async () => {
    console.log('training...')
    setTeaching(true)
    setTeached(false)

    const {
      weights01: newWeights01,
      weights02: newWeights02,
      weights2: newWeights2,
    } = await teacher.teach(
      scaled.arraySync(),
      nu,
      weights01.arraySync(),
      weights02.arraySync(),
      weights2.arraySync(),
      teachOutput.arraySync(),
      count
    )

    setWeights01(tf.tensor(newWeights01))
    setWeights02(tf.tensor(newWeights02))
    setWeights2(tf.tensor(newWeights2))

    setTeaching(false)
    setTeached(true)
    predict(
      tf.tensor(newWeights01),
      tf.tensor(newWeights02),
      tf.tensor(newWeights2)
    )
  }

  const predict = (weights01, weights02, weights2) => {
    let input = tf.concat([scaled, tf.onesLike(teachOutput)], 1)

    let y6 = straightRun(input, weights01, weights02, weights2)

    let counter = 0
    let array = teachOutput.arraySync()

    y6.arraySync().forEach((pr, idx) => {
      let cl = Math.abs(pr - CLASS1) < Math.abs(pr - CLASS2) ? CLASS1 : CLASS2
      let compared =
        array[idx][0] > 0.5
          ? Math.ceil(array[idx][0] * 10) / 10
          : Math.floor(array[idx][0] * 10) / 10

      if (cl === compared) {
        counter++
      }
    })

    alert(`Точность классификации: ${(counter / teachOutput.shape[0]) * 100}%`)
  }

  const straightRun = (input, weights01, weights02, weights2) => {
    //Считаем выходы слоев
    let net4 = tf.dot(input, weights01)
    let net5 = tf.dot(input, weights02)

    let y4 = net4.sigmoid()
    let y5 = net5.sigmoid()

    let hiddenOutput = tf.concat([tf.onesLike(y4), y4, y5], 1)

    let net6 = tf.dot(hiddenOutput, weights2)
    return net6.sigmoid()
  }

  const changeAngle = e => {
    let parsed = parseInt(e)
    if (isNaN(parsed)) {
      return
    }
    setK(parsed)
  }

  const changeQty = e => {
    let parsed = parseInt(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setQty(parsed)
  }

  const changeB = e => {
    let parsed = parseInt(e)
    if (isNaN(parsed)) {
      return
    }
    setB(parsed)
  }

  const changeSd1 = e => {
    let parsed = parseFloat(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setSd1(parsed)
  }

  const changeSd2 = e => {
    let parsed = parseFloat(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setSd2(parsed)
  }

  const changeCount = e => {
    let parsed = parseInt(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setCount(parsed)
  }

  const changeNu = e => {
    let parsed = parseFloat(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setNu(parsed)
  }

  const changeOffset = e => {
    let parsed = parseInt(e)
    if (!e || !parsed || parsed <= 0) {
      return
    }
    setOffset(parsed)
  }

  const resetWeights = () => {
    setWeights01(tf.randomUniform([3, 1], -1, 1))
    setWeights02(tf.randomUniform([3, 1], -1, 1))
    setWeights2(tf.randomUniform([3, 1], -1, 1))
  }

  const buildFromFile = () => {
    let reader = new TxtReader()

    reader.sniffLines(file).then(res => {
      let text = res.result
      let points = []
      let output = []
      let class1 = []
      let class2 = []

      text.forEach(line => {
        let arrayLine = line.split('\t')
        let x = parseFloat(arrayLine[0].replace(',', '.'))
        let y = parseFloat(arrayLine[1].replace(',', '.'))
        if (arrayLine[2]) {
          let cl = parseFloat(arrayLine[2].replace(',', '.'))

          if (cl === CLASS1) {
            class1.push({
              x,
              y,
            })
          } else {
            class2.push({
              x,
              y,
            })
          }

          output.push([cl])
        }

        points.push([x, y])
      })

      setRandomPoints(tf.tensor(points))

      if (class1.length > 0 && class2.length > 0) {
        setFirstClass(class1)
        setSecondClass(class2)
      } else {
        let sc = tf.scalar(1).div(tf.tensor(points))
        let ones = tf.ones([points.length, 1])

        let input = tf.concat([sc, ones], 1)

        input.print()

        let y6 = straightRun(input, weights01, weights02, weights2)

        y6.print()

        y6.arraySync().forEach((result, idx) => {
          let c =
            Math.abs(result - CLASS1) < Math.abs(result - CLASS2)
              ? CLASS1
              : CLASS2

          if (c === CLASS1) {
            class1.push({
              x: points[idx][0],
              y: points[idx][1],
            })
          } else {
            class2.push({
              x: points[idx][0],
              y: points[idx][1],
            })
          }
        })

        setFirstClass(class1)
        setSecondClass(class2)
      }

      setTeachOutput(tf.tensor(output))
    })
  }

  const createFile = () => {
    let output = teachOutput.arraySync()
    let string = ''

    randomPoints.arraySync().forEach((point, idx) => {
      let floored =
        output[idx][0] > 0.5
          ? Math.ceil(output[idx][0] * 10) / 10
          : Math.floor(output[idx][0] * 10) / 10

      string += `${point[0]}\t${point[1]}\t${floored}\n`
    })

    let blob = new Blob([string], {
      type: 'text/plain;charset=utf-8',
    })

    saveAs(blob, `data${new Date().getTime()}.txt`)
  }

  return (
    <Card style={{ marginBottom: 30, padding: 30, border: 'none' }}>
      <Row>
        <Col
          className="d-flex flex-row justify-content-end"
          style={{ marginTop: 15, userSelect: 'none' }}
        >
          <FormGroup className="d-flex flex-row no-wrap" style={{ width: 200 }}>
            <div
              style={{ cursor: 'default', marginRight: 10 }}
              onClick={() => setIsFile(!isFile)}
            >
              Генератор
            </div>
            <CustomInput
              type="switch"
              id="fileSwitch"
              name="fileSwitch"
              checked={isFile}
              label={'Из файла'}
              onChange={() => setIsFile(!isFile)}
            />
          </FormGroup>
        </Col>
      </Row>
      <CardBody className="d-flex flex-column align-items-center">
        {randomPoints && (
          <ScatterChart
            width={500}
            height={500}
            margin={{
              top: 20,
              right: 20,
              bottom: 20,
              left: -30,
            }}
          >
            <CartesianGrid />
            <XAxis type="number" dataKey="x" />
            <YAxis type="number" dataKey="y" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={firstClass} fill="#17a2b7" />
            <Scatter data={secondClass} fill="#dc3546" />
            {/* {!isFile && <Scatter data={line} line fill="blue" />} */}
          </ScatterChart>
        )}
      </CardBody>
      <CardText className="d-flex flex-row justify-content-center">
        {!isFile && (
          <Button style={{ marginRight: 15 }} onClick={generate} color="info">
            Обновить
          </Button>
        )}
        <Button
          style={{ marginRight: 15 }}
          disabled={teaching}
          onClick={fetchWebWorker}
        >
          Обучить
        </Button>
        {!isFile && (
          <Button
            style={{ marginRight: 15 }}
            color="success"
            disabled={!teached}
            onClick={() => predict(weights01, weights02, weights2)}
          >
            Предсказать
          </Button>
        )}
        <Button disabled={teaching} onClick={resetWeights}>
          Сбросить веса
        </Button>
      </CardText>
      <Form style={{ marginBottom: 15 }}>
        {!isFile && (
          <Row form className="d-flex flex-row justify-content-center">
            <Col md={2}>
              <FormGroup>
                <Label for="angle">Угловой коэф-т</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={k}
                  min={-90}
                  onChange={changeAngle}
                />
              </FormGroup>
            </Col>
            <Col md={2}>
              <FormGroup>
                <Label for="qty">Кол-во элементов</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={qty}
                  min={1}
                  onChange={changeQty}
                />
              </FormGroup>
            </Col>
            <Col md={2}>
              <FormGroup>
                <Label for="sd">СКО1</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={sd1}
                  min={0.01}
                  step={0.1}
                  onChange={changeSd1}
                />
              </FormGroup>
            </Col>
            <Col md={2}>
              <FormGroup>
                <Label for="sd">СКО2</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={sd2}
                  min={0.01}
                  step={0.1}
                  onChange={changeSd2}
                />
              </FormGroup>
            </Col>
            <Col md={2}>
              <FormGroup>
                <Label for="sd">Смещение от прямой</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={offset}
                  min={1}
                  step={1}
                  onChange={changeOffset}
                />
              </FormGroup>
            </Col>
          </Row>
        )}
        <Row form className="d-flex flex-row justify-content-center">
          {!isFile && (
            <Col md={2}>
              <FormGroup>
                <Label for="count">B</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={b}
                  min={0}
                  onChange={changeB}
                />
              </FormGroup>
            </Col>
          )}
          <Col md={2}>
            <FormGroup>
              <Label for="count">Кол-во циклов</Label>
              <InputNumber
                style={{ width: '100%' }}
                value={count}
                min={1}
                onChange={changeCount}
              />
            </FormGroup>
          </Col>
          <Col md={2}>
            <FormGroup>
              <Label for="nu">Скорость обучения</Label>
              <InputNumber
                style={{ width: '100%' }}
                value={nu}
                min={0.01}
                step={0.01}
                onChange={changeNu}
              />
            </FormGroup>
          </Col>
          {!isFile && (
            <Col md={2} className="d-flex">
              <Button
                style={{ alignSelf: 'flex-end', marginBottom: '1rem' }}
                onClick={createFile}
              >
                Выгрузить в файл
              </Button>
            </Col>
          )}
        </Row>
        {isFile && (
          <Row form className="d-flex flex-row justify-content-center">
            <Col md={6}>
              <FormGroup>
                <Label for="exampleCustomFileBrowser">Значения из файла</Label>
                <CustomInput
                  id="file"
                  type="file"
                  label="Выберете файл..."
                  accept="text/plain"
                  onChange={e => setFile(e.target.files[0])}
                />
              </FormGroup>
            </Col>
            <Col md={2} className="d-flex">
              <Button
                disabled={!file}
                style={{ alignSelf: 'flex-end', marginBottom: '1rem' }}
                onClick={buildFromFile}
              >
                Построить
              </Button>
            </Col>
          </Row>
        )}
      </Form>
    </Card>
  )
}

export default Generator
