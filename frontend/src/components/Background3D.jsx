import { useEffect, useRef } from 'react'
import * as THREE from 'three'

export default function Background3D({ theme }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    let width = window.innerWidth
    let height = window.innerHeight

    // ── Renderer ──
    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

    // ── Scene & Camera ──
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100)
    camera.position.set(0, 5, 20)
    camera.lookAt(0, 0, 0)

    // ── Color Configs ──
    const isDark = theme === 'dark'
    const barColor = isDark ? 0x6366f1 : 0x2563eb
    const waveColor = isDark ? 0x8b5cf6 : 0x3b82f6
    const particleColor = isDark ? 0x06b6d4 : 0x60a5fa

    // ── 1. Grid of 3D Audio Spectrum Bars ──
    const barCount = 18
    const bars = []
    const barGroup = new THREE.Group()

    const barGeometry = new THREE.BoxGeometry(0.6, 4, 0.6)
    const barMaterial = new THREE.MeshBasicMaterial({
      color: barColor,
      transparent: true,
      opacity: isDark ? 0.22 : 0.12,
      wireframe: true
    })

    for (let i = 0; i < barCount; i++) {
      const bar = new THREE.Mesh(barGeometry, barMaterial)
      // Arrange in a horizontal row across the center floor
      const x = (i - (barCount - 1) / 2) * 1.5
      bar.position.set(x, -2, -5)
      barGroup.add(bar)
      bars.push(bar)
    }
    scene.add(barGroup)

    // ── 2. Dynamic Waveform Ribbon Lines ──
    const ribbonCount = 2
    const ribbons = []
    const pointsCount = 40
    
    for (let r = 0; r < ribbonCount; r++) {
      const ribbonGeometry = new THREE.BufferGeometry()
      const ribbonPositions = new Float32Array(pointsCount * 3)
      
      ribbonGeometry.setAttribute('position', new THREE.BufferAttribute(ribbonPositions, 3))
      
      const ribbonMaterial = new THREE.LineBasicMaterial({
        color: waveColor,
        transparent: true,
        opacity: isDark ? 0.25 : 0.18,
        linewidth: 2
      })
      
      const ribbonLine = new THREE.Line(ribbonGeometry, ribbonMaterial)
      scene.add(ribbonLine)
      ribbons.push({ line: ribbonLine, geom: ribbonGeometry, offset: r * Math.PI })
    }

    // ── 3. Audio Pulse Rings ──
    const ringCount = 3
    const rings = []
    const ringGeometry = new THREE.RingGeometry(0.1, 5, 32)
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: waveColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
      wireframe: true
    })

    for (let i = 0; i < ringCount; i++) {
      const ring = new THREE.Mesh(ringGeometry, ringMaterial.clone())
      ring.rotation.x = Math.PI / 2
      ring.position.set(0, -3, -5)
      scene.add(ring)
      rings.push({ mesh: ring, scale: 0.1 + (i * 0.4), baseOpacity: isDark ? 0.18 : 0.08 })
    }

    // ── 4. Sound Signal Particles ──
    const particleCount = 100
    const pointsGeometry = new THREE.BufferGeometry()
    const positions = new Float32Array(particleCount * 3)
    const originalPositions = []
    const speeds = []

    for (let i = 0; i < particleCount; i++) {
      const x = (Math.random() - 0.5) * 40
      const y = (Math.random() - 0.5) * 20
      const z = (Math.random() - 0.5) * 15

      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z

      originalPositions.push({ x, y, z })
      speeds.push((0.15 + Math.random() * 0.4) * 0.03)
    }

    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

    const pointsMaterial = new THREE.PointsMaterial({
      color: particleColor,
      size: isDark ? 0.25 : 0.35,
      transparent: true,
      opacity: isDark ? 0.4 : 0.25,
      sizeAttenuation: true
    })

    const pointParticles = new THREE.Points(pointsGeometry, pointsMaterial)
    scene.add(pointParticles)

    // ── Mouse & Parallax ──
    let mouseX = 0
    let mouseY = 0
    let targetX = 0
    let targetY = 0

    const handleMouseMove = (e) => {
      mouseX = (e.clientX - window.innerWidth / 2) / 100
      mouseY = (e.clientY - window.innerHeight / 2) / 100
    }
    window.addEventListener('mousemove', handleMouseMove)

    // ── Resize ──
    const handleResize = () => {
      width = window.innerWidth
      height = window.innerHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
    }
    window.addEventListener('resize', handleResize)

    // ── Animation Loop ──
    let animationFrameId
    const clock = new THREE.Clock()

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate)

      const time = clock.getElapsedTime()

      // 1. Animate 3D Audio Bars (simulate sound waves)
      bars.forEach((bar, idx) => {
        // Calculate scaling height using sine/cosine offsets
        const offset = idx * 0.4
        const scaleVal = 0.2 + Math.abs(Math.sin(time * 2.5 + offset)) * 2.8
        bar.scale.y = scaleVal
        // Reposition slightly so they scale upwards from floor
        bar.position.y = -4 + scaleVal * 2
      })

      // 2. Animate Waveform Ribbon Lines
      ribbons.forEach((ribbon) => {
        const positionsArr = ribbon.geom.attributes.position.array
        
        for (let i = 0; i < pointsCount; i++) {
          const t = (i / pointsCount) * Math.PI * 4
          const x = (i - pointsCount / 2) * 0.8
          // Wave oscillation based on sine
          const y = Math.sin(t - time * 3 + ribbon.offset) * 1.5 - 1.5
          const z = Math.cos(t * 0.5 + time) * 2 - 6

          positionsArr[i * 3] = x
          positionsArr[i * 3 + 1] = y
          positionsArr[i * 3 + 2] = z
        }
        ribbon.geom.attributes.position.needsUpdate = true
      })

      // 3. Animate Expanding Pulse Rings
      rings.forEach((ring) => {
        ring.scale += 0.004
        if (ring.scale > 3) {
          ring.scale = 0.1
        }
        // Fade out as they expand
        const fadeFactor = 1.0 - (ring.scale / 3)
        ring.mesh.material.opacity = ring.baseOpacity * fadeFactor
        ring.mesh.scale.set(ring.scale, ring.scale, ring.scale)
      })

      // 4. Animate Sound particles floating around
      const posArr = pointsGeometry.attributes.position.array
      for (let i = 0; i < particleCount; i++) {
        originalPositions[i].y += speeds[i]
        if (originalPositions[i].y > 10) {
          originalPositions[i].y = -10
        }
        posArr[i * 3] = originalPositions[i].x + Math.sin(time + originalPositions[i].y * 0.25) * 0.4
        posArr[i * 3 + 1] = originalPositions[i].y
        posArr[i * 3 + 2] = originalPositions[i].z
      }
      pointsGeometry.attributes.position.needsUpdate = true

      // Parallax easing (lerp)
      targetX += (mouseX - targetX) * 0.05
      targetY += (mouseY - targetY) * 0.05

      // Subtly rotate groups
      barGroup.rotation.y = targetX * 0.1
      barGroup.rotation.x = -targetY * 0.05
      pointParticles.rotation.y = targetX * 0.08
      pointParticles.rotation.x = -targetY * 0.08

      renderer.render(scene, camera)
    }

    animate()

    // ── Cleanup ──
    return () => {
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('resize', handleResize)
      
      barGroup.clear()
      barGeometry.dispose()
      barMaterial.dispose()
      
      ribbons.forEach(r => {
        r.geom.dispose()
        r.line.material.dispose()
      })
      
      ringGeometry.dispose()
      rings.forEach(r => r.mesh.material.dispose())
      
      pointsGeometry.dispose()
      pointsMaterial.dispose()
      
      renderer.dispose()
    }
  }, [theme])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: -10,
      }}
    />
  )
}
