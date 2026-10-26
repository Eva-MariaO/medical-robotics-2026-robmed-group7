# S2RK course assets

Copied unchanged from `/home/joao/ensino/medical-robotics-2026/week-4/`:

- `s2rk_robot.urdf` and `meshes/*.STL`
- `human_liver.obj`

The URDF retains its original attribution and modifications made for the FCUL
Medical Robotics lab. It identifies the source as the HFUT Open Source Surgical
Robot Research Kit, S2RK_Software/S2RK_Simulation/RCM-Simulation/surr-robot/robotrcm,
commit 3b0b801, and states CC BY-NC-SA 4.0 / PolyForm Noncommercial 1.0.0.
The liver mesh is reused from the same course checkout; no additional licence
information was present there. Retain this provenance with the course materials.

The haptics scene holds j1–j5 at HOME and drives the original prismatic j6.
The liver mesh is visual; collision uses a one-sided compliant surface through
the visible entry point, avoiding reverse contact on inner mesh triangles.
Bone is a new capsule primitive. The haptics liver uses softer contact
(300 N/m and 20 N s/m) than week 4 (1500 N/m and 100 N s/m);
bone uses 50000 N/m and 100 N s/m for
an illustrative stiffness comparison. These are teaching models, not identified
patient tissue properties. Mesh deformation and puncture are not simulated.
