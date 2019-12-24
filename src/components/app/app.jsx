import React from 'react'
import Generator from '../generator/generator'
import { Container, Row, Col, Navbar, NavbarBrand } from 'reactstrap'

const App = () => {
  return (
    <>
      <Navbar color="light" light style={{ marginBottom: 30 }}>
        <NavbarBrand style={{ marginLeft: 30 }}>MLP 2-2-1</NavbarBrand>
      </Navbar>
      <Container>
        <Row>
          <Col>
            <Generator />
          </Col>
        </Row>
      </Container>
    </>
  )
}

export default App
