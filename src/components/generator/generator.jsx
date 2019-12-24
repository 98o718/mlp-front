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
// eslint-disable-next-line import/no-webpack-loader-syntax
import teacherWorker from 'workerize-loader!../../teacher.worker'
import InputNumber from 'rc-input-number'
import { TxtReader } from 'txt-reader'
import { saveAs } from 'file-saver'

const Generator = () => {
  const [k, setK] = useState(45)
  const [qty, setQty] = useState(100)
  const [sd, setSd] = useState(2)
  const [count, setCount] = useState(10000)
  const [nu, setNu] = useState(0.5)
  const [scale, setScale] = useState(10)
  const [m, setM] = useState(5)
  const [b, setB] = useState(0)

  const CLASS1 = 0.9
  const CLASS2 = 0.1

  const [randomPoints, setRandomPoints] = useState()
  const [scaled, setScaled] = useState()
  const [teachOutput, setTeachOutput] = useState()
  const [teached, setTeached] = useState(false)
  const [teaching, setTeaching] = useState(false)
  const [teacher, setTeacher] = useState()

  const [file, setFile] = useState()

  const [prediction, setPrediction] = useState()

  const [weights01, setWeights01] = useState()
  const [weights02, setWeights02] = useState()
  const [weights2, setWeights2] = useState()

  tf.initializers.randomUniform({ seed: 1 })

  useEffect(() => {
    setTeacher(teacherWorker())

    setWeights01(tf.randomUniform([3, 1], -1, 1))
    setWeights02(tf.randomUniform([3, 1], -1, 1))
    setWeights2(tf.randomUniform([3, 1], -1, 1))

    let randomNormal = tf.randomNormal([qty, 2], m, sd)

    let output = randomNormal.arraySync().map(point => team(point))

    setRandomPoints(randomNormal)
    setScaled(tf.scalar(1).div(randomNormal))
    setTeachOutput(tf.tensor([output]).transpose())
    // eslint-disable-next-line
  }, [])

  useEffect(() => {
    setRandomPoints(tf.randomNormal([qty, 2], m, sd))
    setTeached(false)
  }, [qty, sd, m])

  useEffect(() => {
    if (randomPoints) {
      let output = randomPoints.arraySync().map(point => team(point))
      setTeached(false)
      tf.scalar(1)
        .div(randomPoints)
        .print()
      setScaled(tf.scalar(1).div(randomPoints))
      setTeachOutput(tf.tensor([output]).transpose())
    }
    // eslint-disable-next-line
  }, [randomPoints, k, b])

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
    predict(
      tf.tensor(newWeights01),
      tf.tensor(newWeights02),
      tf.tensor(newWeights2)
    )
  }

  const refresh = () => {
    setRandomPoints(tf.randomNormal([qty, 2], m, sd))
  }

  const predict = (weights01, weights02, weights2) => {
    let input = tf.concat([scaled, tf.onesLike(teachOutput)], 1)

    //Считаем выходы слоев
    let net4 = tf.dot(input, weights01)
    let net5 = tf.dot(input, weights02)

    let y4 = net4.sigmoid()
    let y5 = net5.sigmoid()

    let hiddenOutput = tf.concat([tf.onesLike(y4), y4, y5], 1)

    let net6 = tf.dot(hiddenOutput, weights2)
    let y6 = net6.sigmoid()

    let counter = 0
    let array = teachOutput.arraySync()

    y6.arraySync().forEach((pr, idx) => {
      let cl = Math.abs(pr - CLASS1) < Math.abs(pr - CLASS2) ? CLASS1 : CLASS2
      let compared =
        array[idx][0] > 0.5
          ? Math.ceil(array[idx][0] * 10) / 10
          : Math.floor(array[idx][0] * 10) / 10

      console.log(cl, compared)
      if (cl === compared) {
        counter++
      }
    })

    alert(
      `Достоверность предсказания: ${(counter / teachOutput.shape[0]) * 100}%`
    )

    setPrediction(y6.arraySync())
    setTeached(true)
  }

  const changeAngle = e => {
    setWeights01(tf.randomUniform([3, 1], -1, 1))
    setWeights02(tf.randomUniform([3, 1], -1, 1))
    setWeights2(tf.randomUniform([3, 1], -1, 1))

    let parsed = parseInt(e)
    if (isNaN(parsed) || parsed < -90 || parsed > 90) {
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

  const changeSd = e => {
    let parsed = parseFloat(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setSd(parsed)
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

  const changeM = e => {
    let parsed = parseFloat(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setM(parsed)
  }

  const changeScale = e => {
    let parsed = parseInt(e)
    if (!e || !parsed || parsed < 0) {
      return
    }
    setScale(parsed)
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

      text.forEach(line => {
        let arrayLine = line.split('\t')
        points.push([
          parseFloat(arrayLine[0].replace(',', '.')),
          parseFloat(arrayLine[1].replace(',', '.')),
        ])
      })

      setRandomPoints(tf.tensor(points))
    })
  }

  const team = point =>
    point[0] * Math.tan(((90 - k) * Math.PI) / 180) + b < point[1] ? 0.9 : 0.1

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
    <Card style={{ marginBottom: 30 }}>
      <CardBody>
        <CardText className="d-flex flex-column align-items-center">
          {randomPoints && (
            <svg
              height="400px"
              width="400px"
              viewBox={`0 0 ${scale} ${scale}`}
              style={{ border: '1px solid grey', transform: 'rotate(270deg)' }}
            >
              {randomPoints.arraySync().map((point, idx) => (
                <circle
                  key={idx}
                  cx={point[0]}
                  cy={point[1]}
                  r="0.1"
                  fill={
                    teached
                      ? Math.abs(prediction[idx] - CLASS1) <
                        Math.abs(prediction[idx] - CLASS2)
                        ? 'magenta'
                        : 'cyan'
                      : team(point) > 0.1
                      ? 'red'
                      : 'blue'
                  }
                />
              ))}
              <line
                x1="-10000"
                y1="0"
                x2="10000"
                y2="0"
                transform={`translate(0, 0) rotate(${90 - k})`}
                stroke="grey"
                strokeWidth="0.05"
              />
            </svg>
          )}
        </CardText>
        <CardText className="d-flex flex-row justify-content-center">
          <Button style={{ marginRight: 15 }} onClick={refresh} color="info">
            Обновить
          </Button>
          <Button
            style={{ marginRight: 15 }}
            disabled={teaching}
            onClick={fetchWebWorker}
          >
            Обучить
          </Button>
          <Button
            style={{ marginRight: 15 }}
            color="success"
            // disabled={!teached}
            onClick={() => predict(weights01, weights02, weights2)}
          >
            Предсказать
          </Button>
          <Button disabled={teaching} onClick={resetWeights}>
            Сбросить веса
          </Button>
        </CardText>
        <Form>
          <Row form className="d-flex flex-row justify-content-center">
            <Col md={2}>
              <FormGroup>
                <Label for="angle">Угол прямой</Label>
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
                <Label for="sd">СКО</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={sd}
                  min={0.01}
                  step={0.01}
                  onChange={changeSd}
                />
              </FormGroup>
            </Col>
            <Col md={2}>
              <FormGroup>
                <Label for="sd">Среднее</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={m}
                  min={0.01}
                  step={0.01}
                  onChange={changeM}
                />
              </FormGroup>
            </Col>
          </Row>
          <Row form className="d-flex flex-row justify-content-center">
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
            <Col md={2}>
              <FormGroup>
                <Label for="nu">Масштаб</Label>
                <InputNumber
                  style={{ width: '100%' }}
                  value={scale}
                  min={1}
                  onChange={changeScale}
                />
              </FormGroup>
            </Col>
          </Row>
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
            <Col md={2} className="d-flex">
              <Button
                style={{ alignSelf: 'flex-end', marginBottom: '1rem' }}
                onClick={createFile}
              >
                Выгрузить в файл
              </Button>
            </Col>
          </Row>
        </Form>
      </CardBody>
    </Card>
  )
}

export default Generator
