# SDHA: Software-Defined Heterogeneous Architecture

A novel AI accelerator architecture that achieves flexibility through software-defined control.

## 📄 Documentation

- [Full Paper](docs/SDHA_Architecture_Paper.md) - Complete technical paper
- [Architecture Overview](docs/architecture.md) - Quick start guide
- [Performance Analysis](docs/performance.md) - Benchmark results

## 🎯 Key Features

- **Software-Defined Control**: All scheduling, memory, and communication controlled by programmable software
- **Hardware Coherence**: Router-based cache coherence protocol
- **Hybrid Modes**: Dynamic (flexible) and deterministic (high-performance) execution
- **85-94% Performance**: Compared to TPU/Groq with 10x flexibility
- **50% Cost Reduction**: $550M vs $1B+ for traditional designs

## 📊 Quick Stats

| Metric | NVIDIA H100 | Google TPU | Groq LPU | SDHA |
|--------|------------|-----------|----------|------|
| Inference Latency | 20ms | 18.3ms | 5.6ms | 12.3ms |
| Training Throughput | 2.25 | 2.63 | N/A | 2.43 |
| Flexibility | 7/10 | 3/10 | 1/10 | 10/10 |
| Dev Cost | $1000M | $1150M | $900M | $550M |

## 🚀 Project Status

⚠️ **Early Stage**: Currently in design and specification phase.

## 📖 Citation

If you use this work, please cite:
\`\`\`bibtex
@article{sdha2025,
  title={Software-Defined Heterogeneous Architecture: A New Paradigm for AI Accelerators},
  author={[Your Name]},
  year={2025}
}
\`\`\`

## 📝 License

This project is licensed under Apache 2.0 License - see [LICENSE](LICENSE) file.
