FROM rust:1.85-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/codex-openai-proxy /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["codex-openai-proxy"]
CMD ["serve"]
