FROM golang:1.23-alpine AS build

WORKDIR /src

COPY go.mod ./
RUN go mod download

COPY . .
RUN go build -o /out/server ./cmd/server

FROM alpine:3.20

WORKDIR /app

COPY --from=build /out/server /app/server

EXPOSE 8080

CMD ["/app/server"]
