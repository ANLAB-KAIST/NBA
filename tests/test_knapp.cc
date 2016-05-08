#include <cstdint>
#include <cstdio>
#include <cassert>
#include <gtest/gtest.h>
#include <scif.h>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/hostutils.hh>
#include <nba/engines/knapp/sharedutils.hh>
#include <nba/engines/knapp/pollring.hh>
#include <nba/engines/knapp/rma.hh>
#if 0
#require <engines/knapp/ctrl.pb.o>
#require <engines/knapp/hostutils.o>
#require <engines/knapp/pollring.o>
#require <engines/knapp/rma.o>
#endif

using namespace nba::knapp;

class RTEEnvironment : public ::testing::Environment {
    int argc;
    char **argv;

public:
    RTEEnvironment(int argc, char **argv)
        : ::testing::Environment(), argc(argc), argv(argv)
    { }

    void SetUp()
    {
        int rc;
        rc = rte_eal_init(argc, argv);
        assert(rc == 0);
    }

    void TearDown()
    {
        return;
    }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    RTEEnvironment *env = new RTEEnvironment(argc, argv);
    ::testing::AddGlobalTestEnvironment(env);
    return RUN_ALL_TESTS();
}

TEST(KnappCommunicationTest, RawPing) {
    std::string msg = "hello world";
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::PING);
    request.mutable_text()->set_msg(msg);

    char buf[1024];
    uint32_t msgsz = request.ByteSize();
    request.SerializeToArray(buf, msgsz);
    rc = scif_send(sock, &msgsz, sizeof(msgsz), SCIF_SEND_BLOCK);
    ASSERT_EQ(sizeof(msgsz), (unsigned) rc);
    rc = scif_send(sock, buf, msgsz, SCIF_SEND_BLOCK);
    ASSERT_EQ(msgsz, (unsigned) rc);

    rc = scif_recv(sock, &msgsz, sizeof(msgsz), SCIF_RECV_BLOCK);
    ASSERT_EQ(sizeof(msgsz), (unsigned) rc);
    rc = scif_recv(sock, buf, msgsz, SCIF_RECV_BLOCK);
    ASSERT_EQ(msgsz, (unsigned) rc);
    response.ParseFromArray(buf, msgsz);

    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_EQ("hello world", response.text().msg());

    scif_close(sock);
}

TEST(KnappCommunicationTest, Ping) {
    std::string msg = "goodbye world";
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::PING);
    request.mutable_text()->set_msg(msg);

    ctrl_invoke(sock, request, response);

    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_EQ("goodbye world", response.text().msg());

    scif_close(sock);
}

TEST(KnappCommunicationTest, InvalidPing) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::PING);
    // missing text.

    ctrl_invoke(sock, request, response);

    EXPECT_EQ(CtrlResponse::INVALID, response.reply());

    scif_close(sock);
}

TEST(KnappMallocTest, Small) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::MALLOC);
    CtrlRequest::MallocParam *m = request.mutable_malloc();
    m->set_size(1024);
    m->set_align(64);
    ctrl_invoke(sock, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_TRUE(response.has_resource());
    EXPECT_NE(0u, response.resource().handle());
    void *ptr = (void *) (uintptr_t) response.resource().handle();

    request.Clear();
    response.Clear();
    request.set_type(CtrlRequest::FREE);
    request.mutable_resource()->set_handle((uintptr_t) ptr);
    ctrl_invoke(sock, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());

    scif_close(sock);
}

TEST(KnappMallocTest, Large) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    void *ptrs[256];

    for (int i = 0; i < 256; i++) {
        request.Clear();
        response.Clear();
        request.set_type(CtrlRequest::MALLOC);
        CtrlRequest::MallocParam *m = request.mutable_malloc();
        m->set_size(16 * 1024 * 1024);
        m->set_align(64);
        ctrl_invoke(sock, request, response);
        EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
        EXPECT_TRUE(response.has_resource());
        EXPECT_NE(0u, response.resource().handle());
        ptrs[i] = (void *) response.resource().handle();
    }

    for (int i = 0; i < 256; i++) {
        request.Clear();
        response.Clear();
        request.set_type(CtrlRequest::FREE);
        request.mutable_resource()->set_handle((uintptr_t) ptrs[i]);
        ctrl_invoke(sock, request, response);
        EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    }

    scif_close(sock);
}

TEST(KnappvDeviceTest, Single) {
    int rc;
    scif_epd_t ctrl_epd = scif_open();
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(ctrl_epd, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;

    request.set_type(CtrlRequest::CREATE_VDEV);
    CtrlRequest::vDeviceInfoParam *v = request.mutable_vdevinfo();
    v->set_num_pcores(2);
    v->set_num_lcores_per_pcore(3);
    v->set_pipeline_depth(32);
    ctrl_invoke(ctrl_epd, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_TRUE(response.has_resource());
    EXPECT_LE(0u, response.resource().handle());
    void *vdev_handle = (void *) response.resource().handle();
    uint32_t vdev_id  = response.resource().id();

    scif_epd_t vdev_data_epd = scif_open();
    ASSERT_NE(SCIF_OPEN_FAILED, vdev_data_epd);
    remote = { 1, get_mic_data_port(vdev_id) };
    rc = scif_connect(vdev_data_epd, &remote);
    ASSERT_LT(0, rc);

    scif_close(vdev_data_epd);

    request.Clear();
    request.set_type(CtrlRequest::DESTROY_VDEV);
    request.mutable_resource()->set_handle((uintptr_t) vdev_handle);
    ctrl_invoke(ctrl_epd, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());

    scif_close(ctrl_epd);
}

TEST(KnappRMATest, H2DWrite) {
    int rc;
    scif_epd_t ctrl_epd = scif_open();
    ASSERT_NE(SCIF_OPEN_FAILED, ctrl_epd);
    struct scif_portID remote = { 1, KNAPP_CTRL_PORT };
    rc = scif_connect(ctrl_epd, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;

    request.set_type(CtrlRequest::CREATE_VDEV);
    CtrlRequest::vDeviceInfoParam *v = request.mutable_vdevinfo();
    v->set_num_pcores(1);
    v->set_num_lcores_per_pcore(4);
    v->set_pipeline_depth(32);
    ctrl_invoke(ctrl_epd, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_TRUE(response.has_resource());
    EXPECT_LE(0u, response.resource().handle());
    void *vdev_handle = (void *) response.resource().handle();
    uint32_t vdev_id  = response.resource().id();

    scif_epd_t vdev_data_epd = scif_open();
    ASSERT_NE(SCIF_OPEN_FAILED, vdev_data_epd);
    remote = { 1, get_mic_data_port(vdev_id) };
    rc = scif_connect(vdev_data_epd, &remote);
    ASSERT_LT(0, rc);

    {
        /* Intentionally reverse the order of buf/ring declaration and API
         * calls to see if ra/peer_ra are set correctly. */
        RMABuffer buf(vdev_data_epd, 4096, 0);
        PollRing ring(vdev_data_epd, 15, 0);

        request.Clear();
        request.set_type(CtrlRequest::CREATE_POLLRING);
        CtrlRequest::PollRingParam *ring_param = request.mutable_pollring();
        ring_param->set_vdev_handle((uintptr_t) vdev_handle);
        ring_param->set_ring_id(0);
        ring_param->set_len(15);
        ring_param->set_local_ra((uint64_t) ring.ra());
        ctrl_invoke(ctrl_epd, request, response);
        EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
        ring.set_peer_ra(response.resource().peer_ra());
        printf("ring: va=%p, ra=%p, peer_ra=%p\n",
               (void*) ring.va(), (void*) ring.ra(), (void*) ring.peer_ra());

        request.Clear();
        request.set_type(CtrlRequest::CREATE_RMABUFFER);
        CtrlRequest::RMABufferParam *rma_param = request.mutable_rma();
        rma_param->set_vdev_handle((uintptr_t) 0); // global
        rma_param->set_buffer_id(0);
        rma_param->set_size(4096);
        rma_param->set_local_ra((uint64_t) buf.ra());
        ctrl_invoke(ctrl_epd, request, response);
        EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
        buf.set_peer_ra(response.resource().peer_ra());
        buf.set_peer_va(response.resource().peer_va());
        printf("rma: va=%p, ra=%p, peer_ra=%p, peer_va=%p\n",
               (void*) buf.va(), (void*) buf.ra(), (void*) buf.peer_ra(), (void*) buf.peer_va());

        ring.notify(0, KNAPP_TERMINATE);
        memset((void *) buf.va(), 1, sizeof(int));
        buf.write(0, sizeof(int), true);
        ring.remote_notify(0, KNAPP_TERMINATE);

        // TODO: take back the RMA buffer and check the content.

        /* Remote poll_rings & rma_buffers are destroyed along with vDevice.
         * Destroying them manually may cause nullptr references in
         * master/worker threads loop that are still running.
         */

        request.Clear();
        request.set_type(CtrlRequest::DESTROY_RMABUFFER);
        request.mutable_rma_ref()->set_vdev_handle((uintptr_t) 0); // global
        request.mutable_rma_ref()->set_buffer_id(0);
        ctrl_invoke(ctrl_epd, request, response);
        assert(CtrlResponse::SUCCESS == response.reply());
    }

    request.Clear();
    request.set_type(CtrlRequest::DESTROY_VDEV);
    request.mutable_resource()->set_handle((uintptr_t) vdev_handle);
    ctrl_invoke(ctrl_epd, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());

    scif_close(ctrl_epd);
}

// vim: ts=8 sts=4 sw=4 et
