#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PCM_FRAME_BYTES 1920u /* 10 ms, 48 kHz, stereo, signed 16-bit LE. */
#define TAP_HEADER_BYTES 40u

struct __attribute__((packed)) tap_header {
    char magic[4];
    uint16_t version_le;
    uint16_t header_bytes_le;
    uint64_t stream_id_le;
    uint64_t sequence_le;
    uint64_t realtime_ns_le;
    uint32_t payload_bytes_le;
    uint32_t flags_le;
};

static uint16_t le16(uint16_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return value;
#else
    return __builtin_bswap16(value);
#endif
}

static uint32_t le32(uint32_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return value;
#else
    return __builtin_bswap32(value);
#endif
}

static uint64_t le64(uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return value;
#else
    return __builtin_bswap64(value);
#endif
}

static int ensure_fifo(const char *path) {
    struct stat status;
    if (mkfifo(path, 0644) == 0) return 0;
    if (errno != EEXIST || stat(path, &status) != 0 || !S_ISFIFO(status.st_mode)) return -1;
    return 0;
}

static int open_nonblocking_writer(const char *path) {
    return open(path, O_WRONLY | O_NONBLOCK | O_CLOEXEC);
}

static void best_effort_write(int fd, const void *data, size_t size) {
    if (fd < 0 || size == 0) return;
    ssize_t written = write(fd, data, size);
    (void)written;
}

static uint64_t realtime_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void publish_tap_frame(const char *path, int *tap_fd, uint64_t stream_id, uint64_t sequence, const uint8_t *pcm, uint32_t size) {
    uint8_t packet[TAP_HEADER_BYTES + PCM_FRAME_BYTES];
    struct tap_header header = {
        .magic = {'O', 'X', 'R', '1'},
        .version_le = le16(1),
        .header_bytes_le = le16(TAP_HEADER_BYTES),
        .stream_id_le = le64(stream_id),
        .sequence_le = le64(sequence),
        .realtime_ns_le = le64(realtime_ns()),
        .payload_bytes_le = le32(size),
        .flags_le = le32(size < PCM_FRAME_BYTES ? 1u : 0u),
    };
    memcpy(packet, &header, sizeof(header));
    memcpy(packet + TAP_HEADER_BYTES, pcm, size);

    if (*tap_fd < 0) *tap_fd = open_nonblocking_writer(path);
    if (*tap_fd < 0) return;
    /* Packet size stays below PIPE_BUF, so a successful write is atomic. */
    ssize_t written = write(*tap_fd, packet, TAP_HEADER_BYTES + size);
    if (written < 0 && (errno == EPIPE || errno == ENXIO)) {
        close(*tap_fd);
        *tap_fd = -1;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s COMPAT_FIFO MAIN_FIFO TAP_FIFO\n", argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    if (ensure_fifo(argv[1]) != 0 || ensure_fifo(argv[2]) != 0 || ensure_fifo(argv[3]) != 0) {
        fprintf(stderr, "audio_fanout: cannot create FIFO: %s\n", strerror(errno));
        return 1;
    }

    /* Keep MAIN_FIFO writable even if the visualization consumer restarts. */
    int main_guard = open(argv[2], O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    int main_fd = open_nonblocking_writer(argv[2]);
    if (main_guard < 0 || main_fd < 0) {
        fprintf(stderr, "audio_fanout: cannot open main FIFO: %s\n", strerror(errno));
        return 1;
    }

    uint8_t frame[PCM_FRAME_BYTES];
    size_t buffered = 0;
    uint64_t sequence = 0;
    uint64_t stream_id = realtime_ns() ^ ((uint64_t)getpid() << 32);
    int tap_fd = -1;
    for (;;) {
        ssize_t count = read(STDIN_FILENO, frame + buffered, PCM_FRAME_BYTES - buffered);
        if (count == 0) break;
        if (count < 0) {
            if (errno == EINTR) continue;
            break;
        }
        best_effort_write(main_fd, frame + buffered, (size_t)count);
        buffered += (size_t)count;
        if (buffered == PCM_FRAME_BYTES) {
            publish_tap_frame(argv[3], &tap_fd, stream_id, sequence++, frame, PCM_FRAME_BYTES);
            buffered = 0;
        }
    }
    if (buffered) publish_tap_frame(argv[3], &tap_fd, stream_id, sequence, frame, (uint32_t)buffered);
    if (tap_fd >= 0) close(tap_fd);
    close(main_fd);
    close(main_guard);
    return 0;
}
