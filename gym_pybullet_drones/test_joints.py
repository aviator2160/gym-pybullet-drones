import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, plane)

stick_extents = [0.1, 0.5, 0.1]
col_stick1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=stick_extents)
col_stick2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=stick_extents)

stick1 = p.createMultiBody(1, col_stick1, -1, [0,0,1], [0,0,0,1])
stick2 = p.createMultiBody(1, col_stick2, -1, [0,1.1,0.9], [0,0,0,1])

constraint = p.createConstraint(stick1, -1, stick2, -1,
                                jointType=p.JOINT_POINT2POINT,
                                jointAxis=[1,0,0],
                                parentFramePosition=[0,0.55,0],
                                childFramePosition=[0,-0.55,0])

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

while (1):
  keys = p.getKeyboardEvents()
  #print(keys)

  time.sleep(0.01)