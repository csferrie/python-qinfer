import qinfer.dialogs as dialogs
from qinfer.tests.base_test import DerandomizedTestCase
import socket
import mock
import multiprocessing.connection

__author__ = 'Michal Kononenko'

class TestDialogs(DerandomizedTestCase):
    pass

class TestPrettyTime(TestDialogs):

    def setUp(self):
        self.long_time = 1e5
        self.assertGreaterEqual(self.long_time, 86400)
        self.assertGreaterEqual(self.long_time - 86400, 3600)
        self.assertGreaterEqual(self.long_time - 86400 - 3600, 60)

        self.short_time = 30

    def test_pretty_time_force_day(self):
        expected_output = r'1 days, 3:46:40'
        self.assertEqual(expected_output, dialogs.pretty_time(self.long_time))

    def test_pretty_time_only_seconds(self):
        expected_output = r'30 seconds'
        self.assertEqual(expected_output, dialogs.pretty_time(self.short_time))


class TestGetConn(TestDialogs):

    def setUp(self):
        TestDialogs.setUp(self)
        self.port = 10000
        self.expected_authkey='notreallysecret'

    mock_listener = mock.MagicMock(spec=multiprocessing.connection.Listener)

    @mock.patch('multiprocessing.connection.Listener',
                return_value=mock_listener)
    def test_get_conn(self, mock_constructor_call):
        listener, port = dialogs._get_conn()

        self.assertEqual(self.mock_listener, listener)
        self.assertEqual(port, self.port)

        expected_call = mock.call(('localhost', int(self.port)),
                                  authkey=self.expected_authkey)
        self.assertEqual(expected_call, mock_constructor_call.call_args)

    def test_get_conn_error_98(self):
        mock_socket_error = socket.error()
        mock_socket_error.errno = 98

        self.assertIsInstance(mock_socket_error, socket.error)

        with mock.patch('multiprocessing.connection.Listener',
                        side_effect=mock_socket_error) as mock_listener_call:
            with self.assertRaises(dialogs.UnableToGetConnError):
                dialogs._get_conn()

        self.assertTrue(mock_listener_call.called)

    def test_get_conn_general_error(self):
        mock_socket_error = socket.error()
        mock_socket_error.errno = 99

        self.assertNotEqual(mock_socket_error.errno, 98)

        with mock.patch('multiprocessing.connection.Listener',
                        side_effect=mock_socket_error) as mock_listener_call:
            with self.assertRaises(socket.error):
                dialogs._get_conn()

        self.assertTrue(mock_listener_call.called)