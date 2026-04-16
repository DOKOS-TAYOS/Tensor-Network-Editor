export function cloneGraphElementDescriptor(descriptor) {
  return {
    classes: descriptor.classes || "",
    data: { ...descriptor.data },
    grabbable: Boolean(descriptor.grabbable),
    group: descriptor.group,
    position: descriptor.position
      ? {
          x: descriptor.position.x,
          y: descriptor.position.y,
        }
      : null,
    selectable: descriptor.selectable !== false,
  };
}

export function graphElementDataEqual(leftData = {}, rightData = {}) {
  const leftKeys = Object.keys(leftData);
  const rightKeys = Object.keys(rightData);
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  return leftKeys.every((key) => leftData[key] === rightData[key]);
}

export function graphElementPositionEqual(leftPosition, rightPosition) {
  if (!leftPosition || !rightPosition) {
    return leftPosition === rightPosition;
  }
  return leftPosition.x === rightPosition.x && leftPosition.y === rightPosition.y;
}

export function graphElementDescriptorsEqual(leftDescriptor, rightDescriptor) {
  if (!leftDescriptor || !rightDescriptor) {
    return leftDescriptor === rightDescriptor;
  }
  return (
    leftDescriptor.group === rightDescriptor.group &&
    leftDescriptor.classes === rightDescriptor.classes &&
    leftDescriptor.selectable === rightDescriptor.selectable &&
    leftDescriptor.grabbable === rightDescriptor.grabbable &&
    graphElementPositionEqual(leftDescriptor.position, rightDescriptor.position) &&
    graphElementDataEqual(leftDescriptor.data, rightDescriptor.data)
  );
}
