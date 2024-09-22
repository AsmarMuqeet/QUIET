// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
ry(1.7092511325680477) q[0];
ry(1.682873047977714) q[1];
ry(1.5108045268930206) q[2];
cx q[2],q[1];
ry(1.035450767474285) q[1];
cx q[2],q[1];
cx q[1],q[0];
ry(0.3177499787899435) q[0];
cx q[2],q[0];
ry(0.15368907928867065) q[0];
cx q[1],q[0];
ry(0.6814046158063358) q[0];
cx q[2],q[0];
ry(5*pi/8) q[3];
cry(-0.19626712795715498) q[0],q[3];
cry(-0.39253425591430996) q[1],q[3];
x q[1];
cry(-0.7850685118286199) q[2],q[3];
cx q[0],q[5];
x q[5];
x q[6];
ccx q[1],q[5],q[6];
ccx q[2],q[6],q[4];
cx q[4],q[3];
u(pi/8,0,0) q[3];
cx q[4],q[3];
u3(pi/8,-pi,-pi) q[3];
cx q[4],q[3];
u(-0.049066781989288745,0,0) q[3];
cx q[4],q[3];
u(0.049066781989288745,0,0) q[3];
x q[6];
ccx q[1],q[5],q[6];
x q[1];
x q[5];
cx q[0],q[5];
ccx q[4],q[0],q[3];
cx q[4],q[3];
u(0.049066781989288745,0,0) q[3];
cx q[4],q[3];
u(-0.049066781989288745,0,0) q[3];
ccx q[4],q[0],q[3];
cx q[0],q[5];
cx q[4],q[3];
u(-0.09813356397857749,0,0) q[3];
cx q[4],q[3];
u(0.09813356397857749,0,0) q[3];
ccx q[4],q[1],q[3];
cx q[4],q[3];
u(0.09813356397857749,0,0) q[3];
cx q[4],q[3];
u(-0.09813356397857749,0,0) q[3];
ccx q[4],q[1],q[3];
x q[1];
cx q[4],q[3];
u(-0.19626712795715498,0,0) q[3];
cx q[4],q[3];
u(0.19626712795715498,0,0) q[3];
ccx q[4],q[2],q[3];
cx q[4],q[3];
u(0.19626712795715498,0,0) q[3];
cx q[4],q[3];
u(-0.19626712795715498,0,0) q[3];
ccx q[4],q[2],q[3];
x q[5];
ccx q[1],q[5],q[6];
x q[6];
ccx q[2],q[6],q[4];
ccx q[1],q[5],q[6];
x q[1];
x q[5];
cx q[0],q[5];
x q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];