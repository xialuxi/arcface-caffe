def get_pose_landmarks(net , img):
    halfheight = img.shape[0] * 0.5
    halfwidth = img.shape[1] * 0.5
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_LINEAR)
    img = (img - 127.5) * 0.0078125
    x = np.array(img)
    x = np.transpose(x, (2, 0, 1))
    net.blobs['data'].data[...][0] = x
    out = net.forward()

    pose = out['fc2'][0][0:3]
    landmark = out['fc2'][0][3:]

    pose = pose * 90.0
    landmarks = []
    for i in range(21):
        point = []
        point.append(landmark[i * 2 + 0] * halfwidth + halfwidth)
        point.append(landmark[i * 2 + 1] * halfheight + halfheight)
        landmarks.append(point)

    landmarks = np.array(landmarks, dtype=np.int32)

    return pose, landmarks
