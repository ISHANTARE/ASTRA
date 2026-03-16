import { useRef, useMemo, useEffect } from "react";
import * as THREE from "three";

// Color map: satellite=green, debris=red, rocket_body=gray
const TYPE_COLORS = {
  satellite: new THREE.Color(0x00e676),
  debris: new THREE.Color(0xff4c6a),
  rocket_body: new THREE.Color(0x8892a8),
  unknown: new THREE.Color(0x5a6478),
};

/**
 * Renders 30,000+ orbital objects using InstancedMesh for performance.
 * Positions are provided directly from the backend SGP4 propagator.
 * 
 * Per doc 05: Use InstancedMesh in Three.js to render 30,000+ objects smoothly.
 * Color coding: Active Sats = Green, Debris = Red, Rocket Bodies = Gray.
 */
export default function SatelliteRenderer({ objects = [], positions = [] }) {
  const meshRef = useRef();
  const count = objects.length;

  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Compute colors
  const colorArray = useMemo(() => {
    // If no objects, return empty
    if (count === 0) return new Float32Array();
    const colors = new Float32Array(count * 3);
    objects.forEach((obj, i) => {
      const color = TYPE_COLORS[obj.type || "unknown"] || TYPE_COLORS.unknown;
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });
    return colors;
  }, [objects, count]);

  // Set instance colors
  useEffect(() => {
    if (!meshRef.current || count === 0) return;
    const mesh = meshRef.current;
    mesh.geometry.setAttribute(
      "color",
      new THREE.InstancedBufferAttribute(colorArray, 3)
    );
    mesh.material.vertexColors = true;
    mesh.material.needsUpdate = true;
  }, [colorArray, count]);

  // Update positions from backend data
  useEffect(() => {
    if (!meshRef.current || count === 0 || positions.length === 0) return;
    
    // Safety check: positions should be 3x the number of objects
    if (positions.length < count * 3) {
      console.warn("Received positions array is smaller than expected");
      return;
    }

    for (let i = 0; i < count; i++) {
        // positions array is [x1, y1, z1, x2, y2, z2, ...]
        const x = positions[i * 3];
        const y = positions[i * 3 + 1];
        const z = positions[i * 3 + 2];

        // The backend teme_to_visualization already applies the Y-up mapping, 
        // so we can use x, y, z directly here.
        dummy.position.set(x, y, z);
        dummy.scale.setScalar(1);
        dummy.updateMatrix();
        meshRef.current.setMatrixAt(i, dummy.matrix);
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [positions, count]);

  if (count === 0) return null;

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} frustumCulled={false}>
      <sphereGeometry args={[0.004, 4, 4]} />
      <meshBasicMaterial transparent opacity={0.85} />
    </instancedMesh>
  );
}
