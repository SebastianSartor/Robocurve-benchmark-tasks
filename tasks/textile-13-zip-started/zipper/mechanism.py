"""Shared zipper geometry and passive joints for the two TextileBench pilots."""

LENGTH = 0.180
STARTED = 0.020
PIN_TRAVEL = 0.038
PITCH = 0.006
END_TOLERANCE = 0.003
PIN_SEATED_TOLERANCE = 0.003
CONTROL_HZ = 20
MAX_STEPS = 240 * CONTROL_HZ
ORIGIN_XY = (0.145, 0.0)

# Each box is (center, size) in its moving link's local frame, in meters.
# The gate's right edge is at -9 mm when the slider is at the bottom.
GATE = ((-0.114, -0.027, 0.018), (0.210, 0.006, 0.007))
PIN_BOSS = ((0.0, -0.018, 0.018), (0.008, 0.008, 0.007))
SLIDER_HANDLE = ((0.0, 0.0, 0.035), (0.018, 0.016, 0.024))
PIN_HANDLE = ((0.004, -0.061, 0.034), (0.018, 0.016, 0.024))
BRACE_HANDLE = ((0.042, 0.073, 0.029), (0.028, 0.018, 0.028))

BLUE = (0.055, 0.20, 0.36)
TAPE = (0.08, 0.31, 0.43)
STEEL = (0.64, 0.68, 0.74)
ORANGE = (0.95, 0.31, 0.05)
YELLOW = (0.98, 0.77, 0.10)


def author_mechanism(stage, path, translation, orientation, initial_slider, *, auxiliary_grips=True):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    from .visual import author_row

    root = UsdGeom.Xform.Define(stage, path)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        # Isaac Lab spawners supply the quaternion in xyzw order.
        root.AddOrientOp().Set(Gf.Quatf(orientation[3], Gf.Vec3f(*orientation[:3])))
    path = root.GetPath()
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)

    friction = UsdShade.Material.Define(stage, path.AppendChild("GripMaterial"))
    physics_material = UsdPhysics.MaterialAPI.Apply(friction.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.9)
    physics_material.CreateDynamicFrictionAttr(0.75)

    def link(name, mass, xyz=(0.0, 0.0, 0.0)):
        body = UsdGeom.Xform.Define(stage, path.AppendChild(name))
        body.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
        return body.GetPath()

    def box(parent, name, geometry, color, *, collision=True, hidden=False):
        center, size = geometry
        cube = UsdGeom.Cube.Define(stage, parent.AppendChild(name))
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collision:
            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(friction, materialPurpose="physics")
        if hidden:
            cube.MakeInvisible()

    base = link("Base", 0.45)
    slider = link("Slider", 0.030, (initial_slider, 0.0, 0.0))
    pin = link("PinSide", 0.12)

    for body, sign in ((base, 1), (pin, -1)):
        box(body, "RailSupport", ((0.087, sign * 0.011, 0.006), (0.221, 0.015, 0.010)), TAPE,
            hidden=True)
        author_row(stage, body, sign, initial_slider)
    box(base, "HemConnector", ((0.0, 0.009, 0.016), (0.022, 0.016, 0.011)), STEEL, collision=False)
    box(slider, "SliderHousing", ((0.0, 0.0, 0.021), (0.018, 0.021, 0.010)), STEEL)
    box(slider, "PullTab", SLIDER_HANDLE, ORANGE)
    box(slider, "Interlock", GATE, STEEL, hidden=True)
    box(pin, "PinBoss", PIN_BOSS, YELLOW, hidden=not auxiliary_grips)
    if auxiliary_grips:
        box(base, "BraceFoot", ((*BRACE_HANDLE[0][:2], 0.008), (0.028, 0.018, 0.014)), BLUE)
        box(base, "BraceGrip", BRACE_HANDLE, BLUE)
        box(pin, "PinFoot", ((*PIN_HANDLE[0][:2], 0.0115), (0.018, 0.016, 0.021)), YELLOW)
        box(pin, "PinGrip", PIN_HANDLE, YELLOW)

    def prismatic(name, child, axis, low, high, resistance):
        joint = UsdPhysics.PrismaticJoint.Define(stage, path.AppendChild(name))
        joint.CreateBody0Rel().SetTargets([base])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(low)
        joint.CreateUpperLimitAttr(high)
        # Only the two sibling links need self-contact for the latch.
        joint.CreateCollisionEnabledAttr(False)
        # Passive friction keeps the hardware from creeping as the cloth settles.
        prim = joint.GetPrim()
        prim.AddAppliedSchema("PhysxJointAxisAPI:linear")
        for setting, value in (
            ("staticFrictionEffort", resistance),
            ("dynamicFrictionEffort", resistance * 0.6),
            ("viscousFrictionCoefficient", 0.5),
        ):
            prim.CreateAttribute(f"physxJointAxis:linear:{setting}", Sdf.ValueTypeNames.Float).Set(value)

    prismatic("slider_travel", slider, "X", 0.0, LENGTH, 0.7)
    prismatic("pin_release", pin, "Y", -PIN_TRAVEL, 0.0, 0.35)
    return root.GetPrim()
