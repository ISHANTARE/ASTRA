import { useRef, useMemo } from "react";
import { useFrame, useLoader } from "@react-three/fiber";
import * as THREE from "three";

/**
 * 3D Earth Globe with atmosphere glow.
 * Uses a basic sphere with procedural textures for a clean look.
 */
export default function Globe() {
  const earthRef = useRef();
  const atmosphereRef = useRef();

  // Slow rotation for visual effect
  useFrame((_, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.02;
    }
  });

  // Earth material with a deep blue/green tint
  const earthMaterial = useMemo(() => {
    return new THREE.MeshPhongMaterial({
      color: new THREE.Color(0x1a3a5c),
      emissive: new THREE.Color(0x071525),
      emissiveIntensity: 0.3,
      shininess: 25,
      transparent: false,
    });
  }, []);

  // Atmosphere glow shader material
  const atmosphereMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.5);
          gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
    });
  }, []);

  return (
    <group>
      {/* Earth sphere */}
      <mesh ref={earthRef} material={earthMaterial}>
        <sphereGeometry args={[1, 64, 64]} />
      </mesh>

      {/* Wireframe overlay for visual detail */}
      <mesh rotation={[0, 0, 0]}>
        <sphereGeometry args={[1.001, 36, 18]} />
        <meshBasicMaterial
          color={0x3a6aaa}
          wireframe
          transparent
          opacity={0.06}
        />
      </mesh>

      {/* Atmosphere glow */}
      <mesh ref={atmosphereRef} scale={[1.15, 1.15, 1.15]} material={atmosphereMaterial}>
        <sphereGeometry args={[1, 64, 64]} />
      </mesh>

      {/* Ambient light */}
      <ambientLight intensity={0.2} />

      {/* Sun-like directional light */}
      <directionalLight
        position={[5, 3, 5]}
        intensity={1.2}
        color={0xfff5e6}
      />
    </group>
  );
}
