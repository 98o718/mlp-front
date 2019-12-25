import React from 'react'
import Generator from '../generator/generator'
import { Container, Row, Col } from 'reactstrap'

const App = () => {
  return (
    <Container>
      <Row>
        <Col>
          <Generator />
        </Col>
      </Row>
    </Container>
  )
}

export default App
